/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include <windows.h>

#include "rtx_ngx_neural_rendering.h"

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>

#include <vulkan/vulkan.h>

#include "rtx_resources.h"
#include "rtx_options.h"

#include <dxvk_context.h>
#include <dxvk_device.h>
#include <dxvk_scoped_annotation.h>

#include "../util/log/log.h"
#include "../util/util_env.h"
#include "../util/util_once.h"
#include "../util/util_string.h"

#include <cstdio>

namespace dxvk {
  namespace {
    std::string resultToString(NVSDK_NGX_Result result) {
      char buf[1024];
      snprintf(buf, sizeof(buf), "(code: 0x%08x, info: %ls)", result, GetNGXResultAsString(result));
      buf[sizeof(buf) - 1] = '\0';
      return std::string(buf);
    }

    // NGX feature id for DLSS Neural Rendering. Not in the public SDK enum, so it is spelled
    // out. Confirmed against nvngx_dlssnr.dll (DLSSNR 310.8.0): every *_GetFeatureRequirements
    // builds its request with `mov dword [rsp+X], 0x12`.
    constexpr NVSDK_NGX_Feature kFeatureDLSSNR = static_cast<NVSDK_NGX_Feature>(18);

    // The snippet ships exactly one network. Its weight registry is one entry wide
    // (base 0x1800b0d80, end 0x1800b1008, stride 0x288) and the accessor hardcodes `cmp rcx,1`.
    // That entry declares preset 1, config "CC_SILVER_AARDWOLD". Any other value logs a
    // fallback and loads the same weights, so there is nothing to choose between and no preset
    // selector is exposed anywhere in this integration.
    constexpr unsigned int kOnlyPreset = 1;

    // Parameter names, read out of the snippet's own string table.
    constexpr const char* kParamColor = "DLSSNR.Color";
    constexpr const char* kParamDepth = "DLSSNR.Depth";
    constexpr const char* kParamMVec = "DLSSNR.MVec";
    constexpr const char* kParamOutput = "DLSSNR.Output";
    constexpr const char* kParamControlMask = "DLSSNR.ControlMask";

    constexpr const char* kParamWidth = "DLSSNR.Width";
    constexpr const char* kParamHeight = "DLSSNR.Height";
    // NV-DXVK start: DLSS-NR
    // Note: DLSSNR.InputWidth/InputHeight are INERT with this snippet --- neither string exists
    // anywhere in nvngx_dlssnr.dll, and CreateFeature reads only DLSSNR.Width/DLSSNR.Height
    // ("CreateFeature begin requested resolution %ux%u (network %ux%u)" is fed from that one
    // pair, twice). They are still set, and still set to the colour grid, because they are the
    // documented spelling for "the resource bound as DLSSNR.Color" and a future snippet build
    // may start reading them --- but no behaviour may be predicated on them.
    constexpr const char* kParamInputWidth = "DLSSNR.InputWidth";
    constexpr const char* kParamInputHeight = "DLSSNR.InputHeight";
    // NV-DXVK end
    constexpr const char* kParamEnabled = "DLSSNR.Enabled";
    constexpr const char* kParamReset = "DLSSNR.Reset";
    constexpr const char* kParamDepthInverted = "DLSSNR.DepthInverted";
    constexpr const char* kParamMVecScaleX = "DLSSNR.MVecScaleX";
    constexpr const char* kParamMVecScaleY = "DLSSNR.MVecScaleY";
    constexpr const char* kParamScalingRatio = "DLSSNR.ScalingRatio";
    constexpr const char* kParamRenderPreset = "DLSSNR.Hint.Render.Preset";
    constexpr const char* kParamUseAutoMask = "DLSSNR.UseAutoMask";

    constexpr const char* kParamIntensity = "DLSSNR.Intensity";
    constexpr const char* kParamLocalToneStrength = "DLSSNR.LocalToneStrength";
    constexpr const char* kParamLocalStructureStrength = "DLSSNR.LocalStructureStrength";
    constexpr const char* kParamSkinStructureStrength = "DLSSNR.SkinStructureStrength";
    constexpr const char* kParamStyle = "DLSSNR.Style";

    // Subrect suffixes are "<Resource>SubrectBaseX" etc, note there is no dot before Subrect,
    // unlike the "." style the DLSS and DLSS-RR parameter names use.
    //
    // These are built once and kept for the lifetime of the process rather than formatted into a
    // stack buffer per frame: NVSDK_NGX_Parameter::Set takes the name as a bare const char* and
    // nothing in the ABI promises the implementation copies it before returning.
    struct ResourceParamNames {
      explicit ResourceParamNames(const char* name)
        : resource(name)
        , subrectBaseX(std::string(name) + "SubrectBaseX")
        , subrectBaseY(std::string(name) + "SubrectBaseY")
        , subrectWidth(std::string(name) + "SubrectWidth")
        , subrectHeight(std::string(name) + "SubrectHeight") {
      }

      std::string resource;
      std::string subrectBaseX;
      std::string subrectBaseY;
      std::string subrectWidth;
      std::string subrectHeight;
    };

    constexpr const wchar_t* kSnippetModuleName = L"nvngx_dlssnr.dll";
    // See NgxNeuralRenderingSnippet::load() for why this module exists and what its name has
    // to look like.
    constexpr const wchar_t* kTrampolineModuleName = L"remix_nvngx.dll";
    constexpr const wchar_t* kNgxRuntimeModuleName = L"nvngx.dll";

    // Note: this deliberately differs from ViewToResourceVK in rtx_ngx_wrapper.cpp:68, which
    // reports the IMAGE format. For a cross-format-aliased Resources::AliasedResource the image
    // and view formats differ (m_primaryDisocclusionMaskForRR is an R32_SFLOAT view over an
    // R32_UINT image), and the snippet must be told the format it will actually sample through.
    // This matches how the XeSS integration extracts handles (rtx_xess.cpp:578).
    NVSDK_NGX_Resource_VK viewToResourceVK(const Rc<DxvkImageView>& view, bool isUAV) {
      const VkImageView imageView = view->handle();
      const VkImage image = view->imageHandle();
      const VkImageSubresourceRange subresourceRange = view->subresources();
      const VkFormat format = view->info().format;
      const VkExtent3D extent = view->imageInfo().extent;

      return NVSDK_NGX_Create_ImageView_Resource_VK(imageView, image, subresourceRange, format, extent.width, extent.height, isUAV);
    }

    NVSDK_NGX_Resource_VK textureToResourceVK(const Resources::Resource* tex, bool isUAV) {
      if (tex == nullptr || tex->view == nullptr || tex->image == nullptr) {
        return {};
      }

      return viewToResourceVK(tex->view, isUAV);
    }

    // The snippet's own Vulkan exports. Signatures verified against the export table and the
    // disassembly of nvngx_dlssnr.dll (DLSSNR 310.8.0), not just against the public SDK header.
    typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_NgxVulkanInitExt)(unsigned long long appId, const wchar_t* applicationDataPath,
                                                               VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                                                               NVSDK_NGX_Version sdkVersion, const void* featureCommonInfo);
    typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_NgxVulkanShutdown1)(VkDevice device);
    typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_NgxVulkanAllocateParameters)(NVSDK_NGX_Parameter** outParameters);
    typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_NgxVulkanDestroyParameters)(NVSDK_NGX_Parameter* parameters);
    typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_NgxVulkanCreateFeature)(VkCommandBuffer commandBuffer, NVSDK_NGX_Feature featureId,
                                                                     NVSDK_NGX_Parameter* parameters, NVSDK_NGX_Handle** outHandle);
    typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_NgxVulkanReleaseFeature)(NVSDK_NGX_Handle* handle);
    typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_NgxVulkanEvaluateFeature)(VkCommandBuffer commandBuffer, const NVSDK_NGX_Handle* handle,
                                                                       const NVSDK_NGX_Parameter* parameters, void* progressCallback);

    typedef void(NVSDK_CONV* PFN_RemixNgxTrampolineSetSnippet)(void* snippetModule);

    /**
     * Loads nvngx_dlssnr.dll by hand and resolves its Vulkan exports. One instance per process.
     *
     * Why not go through the driver:
     *   nvngx.dll verifies the snippet's Authenticode signature and enforces its
     *   NGXMinimumDriverVersion (615.00) and NGXGpuArchitecture (Blackwell2) resource strings.
     *   A snippet patched to run on Ada fails all three.
     *
     * The supported path is the direct one:
     *   The snippet exports the whole NGX Vulkan surface itself --- Init_Ext,
     *   AllocateParameters, CreateFeature, EvaluateFeature, ReleaseFeature, DestroyParameters,
     *   Shutdown1, GetFeatureRequirements --- so LoadLibraryW plus GetProcAddress is enough to
     *   drive it, and that is the configuration DLSS-NR is known to run in (RTX 4090, driver
     *   610.43.02). It is the same direct-load pattern RenoDX uses on D3D12. Nothing here is
     *   allowed to depend on anything else being present.
     *
     * The optional trampoline (kTrampolineModuleName):
     *   Disassembly of this snippet build shows several entry points opening with
     *     GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | UNCHANGED_REFCOUNT,
     *                        <return address>, &callerModule);
     *     GetModuleFileNameW(callerModule, path, MAX_PATH);
     *     if (!wcsstr(path, L"nvngx.dll")) { log("Error: Not called from NGX runtime - %S");
     *                                        return NVSDK_NGX_Result_FAIL_PlatformError; }
     *   i.e. a case-sensitive substring test on the CALLER module's path. That gate has not
     *   been observed to fire in the configuration above, but if a snippet build does enforce
     *   it, the calls have to be issued from a module whose path contains "nvngx.dll".
     *   remix_nvngx.dll (src/dlssnr_shim/remix_nvngx.cpp) is that module: it exports the same
     *   NVSDK_NGX_VULKAN_* names and forwards each one into the snippet, so it supplies a
     *   return address such a build would accept.
     *
     *   The trampoline is used when it happens to sit beside the runtime and is skipped
     *   silently when it does not. It is never a precondition: a deployment that ships only
     *   nvngx_dlssnr.dll is the one known to work, and must keep working.
     *
     *   NVSDK_NGX_VULKAN_Init (the non-Ext one) is a two instruction stub that returns a
     *   failure code and must not be used.
     *
     *   If d3d9.dll itself is deployed at a path that already contains "nvngx.dll" the
     *   trampoline is not looked for at all.
     */
    class NgxNeuralRenderingSnippet {
    public:
      static NgxNeuralRenderingSnippet& get() {
        static NgxNeuralRenderingSnippet s_snippet;
        return s_snippet;
      }

      bool isAvailable() const {
        return m_available;
      }

      const std::string& getNotAvailableReason() const {
        return m_notAvailableReason;
      }

      PFN_NgxVulkanInitExt initExt = nullptr;
      PFN_NgxVulkanShutdown1 shutdown1 = nullptr;
      PFN_NgxVulkanCreateFeature createFeature = nullptr;
      PFN_NgxVulkanReleaseFeature releaseFeature = nullptr;
      PFN_NgxVulkanEvaluateFeature evaluateFeature = nullptr;
      // Note: these two may legitimately stay null. Every other pointer above is verified to
      // exist before the snippet is reported as available; the parameter block has a fallback
      // (see NGXNeuralRenderingContext::initializeSnippet), so a snippet build without them
      // still runs.
      PFN_NgxVulkanAllocateParameters allocateParameters = nullptr;
      PFN_NgxVulkanDestroyParameters destroyParameters = nullptr;

    private:
      NgxNeuralRenderingSnippet() {
        load();
      }

      // Returns the directory the calling module (d3d9.dll) was loaded from, with a trailing
      // separator, or an empty string when it cannot be determined. Note this is deliberately
      // the module directory rather than env::getExePath(), because the snippet ships beside
      // d3d9.dll the same way nvngx_dlss.dll and nvngx_dlssd.dll do.
      static std::wstring getOwnModulePath() {
        HMODULE ownModule = nullptr;

        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCWSTR>(&NgxNeuralRenderingSnippet::getOwnModulePath),
                                &ownModule)) {
          return std::wstring();
        }

        wchar_t path[MAX_PATH + 1] = {};
        const DWORD length = GetModuleFileNameW(ownModule, path, MAX_PATH);

        if (length == 0 || length > MAX_PATH) {
          return std::wstring();
        }

        return std::wstring(path, length);
      }

      static std::wstring getDirectory(const std::wstring& path) {
        const size_t separator = path.find_last_of(L"\\/");

        if (separator == std::wstring::npos) {
          return std::wstring();
        }

        return path.substr(0, separator + 1);
      }

      void load() {
        const std::wstring ownModulePath = getOwnModulePath();

        if (ownModulePath.empty()) {
          m_notAvailableReason = "Unable to determine the Remix module path.";
          return;
        }

        const std::wstring moduleDirectory = getDirectory(ownModulePath);
        const std::wstring snippetPath = moduleDirectory + kSnippetModuleName;

        m_snippetModule = LoadLibraryW(snippetPath.c_str());

        if (m_snippetModule == nullptr) {
          // Note: the snippet being absent is the expected case for a stock install. Keep this
          // at info level and leave the renderer alone --- DLSS-NR simply does not exist here.
          m_notAvailableReason = str::format(kSnippetModuleName, " was not found next to ", moduleDirectory.c_str(), ".");
          Logger::info(str::format("NVIDIA DLSS-NR not available: ", m_notAvailableReason));
          return;
        }

        // Check the SNIPPET's own export table, not whatever module the calls end up being
        // issued through: the trampoline exports these names unconditionally and ignores a
        // failed GetProcAddress on the snippet, so resolving through it would report a
        // truncated or incompatible nvngx_dlssnr.dll as perfectly fine.
        static const char* const kRequiredSnippetExports[] = {
          "NVSDK_NGX_VULKAN_Init_Ext",
          "NVSDK_NGX_VULKAN_Shutdown1",
          "NVSDK_NGX_VULKAN_CreateFeature",
          "NVSDK_NGX_VULKAN_ReleaseFeature",
          "NVSDK_NGX_VULKAN_EvaluateFeature",
        };

        for (const char* exportName : kRequiredSnippetExports) {
          if (GetProcAddress(m_snippetModule, exportName) == nullptr) {
            m_notAvailableReason = str::format("The DLSS-NR snippet does not export ", exportName, ".");
            Logger::err(str::format("NVIDIA DLSS-NR not available: ", m_notAvailableReason));
            unload();
            return;
          }
        }

        // Pick the module the snippet will see as its caller. The snippet itself is the
        // default and always works; the trampoline is an optional belt-and-braces path for a
        // snippet build that enforces the "called from NGX runtime" gate. Its absence is never
        // a reason to disable the feature.
        HMODULE callModule = m_snippetModule;

        if (ownModulePath.find(kNgxRuntimeModuleName) == std::wstring::npos) {
          const std::wstring trampolinePath = moduleDirectory + kTrampolineModuleName;

          m_trampolineModule = LoadLibraryW(trampolinePath.c_str());

          if (m_trampolineModule != nullptr) {
            const auto setSnippet = reinterpret_cast<PFN_RemixNgxTrampolineSetSnippet>(
              GetProcAddress(m_trampolineModule, "RemixNgxTrampoline_SetSnippet"));

            if (setSnippet != nullptr) {
              setSnippet(m_snippetModule);
              callModule = m_trampolineModule;
            } else {
              Logger::warn(str::format(kTrampolineModuleName, " does not export RemixNgxTrampoline_SetSnippet; "
                                       "calling the DLSS-NR snippet directly instead."));
              FreeLibrary(m_trampolineModule);
              m_trampolineModule = nullptr;
            }
          } else {
            Logger::info(str::format(kTrampolineModuleName, " was not found next to ", moduleDirectory.c_str(),
                                     "; calling the DLSS-NR snippet directly. If the snippet turns out to reject "
                                     "these calls with FAIL_PlatformError (\"Not called from NGX runtime\"), ship "
                                     "the trampoline beside the runtime."));
          }
        }

        // Resolve through the chosen call module, falling back to the snippet for anything the
        // trampoline does not forward (an older trampoline build has no parameter forwarders).
        auto resolve = [this, callModule](const char* name) -> FARPROC {
          FARPROC address = GetProcAddress(callModule, name);

          if (address == nullptr && callModule != m_snippetModule) {
            address = GetProcAddress(m_snippetModule, name);
          }

          return address;
        };

        initExt = reinterpret_cast<PFN_NgxVulkanInitExt>(resolve("NVSDK_NGX_VULKAN_Init_Ext"));
        shutdown1 = reinterpret_cast<PFN_NgxVulkanShutdown1>(resolve("NVSDK_NGX_VULKAN_Shutdown1"));
        createFeature = reinterpret_cast<PFN_NgxVulkanCreateFeature>(resolve("NVSDK_NGX_VULKAN_CreateFeature"));
        releaseFeature = reinterpret_cast<PFN_NgxVulkanReleaseFeature>(resolve("NVSDK_NGX_VULKAN_ReleaseFeature"));
        evaluateFeature = reinterpret_cast<PFN_NgxVulkanEvaluateFeature>(resolve("NVSDK_NGX_VULKAN_EvaluateFeature"));

        // The parameter entry points are only claimed when the SNIPPET has them: the trampoline
        // exports these names whether or not it could forward them, so asking it would hide a
        // snippet build that has no parameter surface behind a stub that always fails.
        const bool snippetHasParameterEntryPoints =
          GetProcAddress(m_snippetModule, "NVSDK_NGX_VULKAN_AllocateParameters") != nullptr &&
          GetProcAddress(m_snippetModule, "NVSDK_NGX_VULKAN_DestroyParameters") != nullptr;

        if (snippetHasParameterEntryPoints) {
          allocateParameters = reinterpret_cast<PFN_NgxVulkanAllocateParameters>(resolve("NVSDK_NGX_VULKAN_AllocateParameters"));
          destroyParameters = reinterpret_cast<PFN_NgxVulkanDestroyParameters>(resolve("NVSDK_NGX_VULKAN_DestroyParameters"));
        }

        // Defensive: the snippet was probed above and resolve() falls back to it, so this can
        // only trigger if the trampoline exported a name it could not actually produce.
        if (initExt == nullptr || shutdown1 == nullptr || createFeature == nullptr ||
            releaseFeature == nullptr || evaluateFeature == nullptr) {
          m_notAvailableReason = "The DLSS-NR snippet does not export the full NVSDK_NGX_VULKAN_* surface.";
          Logger::err(str::format("NVIDIA DLSS-NR not available: ", m_notAvailableReason));
          unload();
          return;
        }

        if (allocateParameters == nullptr || destroyParameters == nullptr) {
          Logger::info("The DLSS-NR snippet exports no AllocateParameters/DestroyParameters; "
                       "the parameter block will come from the linked NGX SDK instead.");
        }

        m_available = true;

        Logger::info(str::format("NVIDIA DLSS-NR snippet loaded from ", snippetPath.c_str(),
                                 (m_trampolineModule != nullptr ? " (via remix_nvngx.dll)" : " (direct)")));
      }

      void unload() {
        initExt = nullptr;
        shutdown1 = nullptr;
        createFeature = nullptr;
        releaseFeature = nullptr;
        evaluateFeature = nullptr;
        allocateParameters = nullptr;
        destroyParameters = nullptr;

        if (m_trampolineModule != nullptr) {
          FreeLibrary(m_trampolineModule);
          m_trampolineModule = nullptr;
        }

        if (m_snippetModule != nullptr) {
          FreeLibrary(m_snippetModule);
          m_snippetModule = nullptr;
        }

        m_available = false;
      }

      HMODULE m_snippetModule = nullptr;
      HMODULE m_trampolineModule = nullptr;
      bool m_available = false;
      std::string m_notAvailableReason;
    };
  }

  bool NGXNeuralRenderingContext::isSnippetAvailable() {
    return NgxNeuralRenderingSnippet::get().isAvailable();
  }

  const std::string& NGXNeuralRenderingContext::getSnippetNotAvailableReason() {
    return NgxNeuralRenderingSnippet::get().getNotAvailableReason();
  }

  std::unique_ptr<NGXNeuralRenderingContext> NGXNeuralRenderingContext::createNeuralRenderingContext(DxvkDevice* device) {
    if (!isSnippetAvailable()) {
      return nullptr;
    }

    auto context = std::make_unique<NGXNeuralRenderingContext>(device);

    if (!context->initializeSnippet()) {
      return nullptr;
    }

    return context;
  }

  NGXNeuralRenderingContext::NGXNeuralRenderingContext(DxvkDevice* device)
    : m_device(device) {
  }

  NGXNeuralRenderingContext::~NGXNeuralRenderingContext() {
    releaseNGXFeature();

    NgxNeuralRenderingSnippet& snippet = NgxNeuralRenderingSnippet::get();

    if (m_parameters) {
      // Release it through whichever side allocated it --- see initializeSnippet(). Never cross
      // the two: a block the snippet allocated must not be handed to the SDK's destroy, and vice
      // versa. The pair is resolved together, so the leak branch below cannot normally happen.
      if (m_parametersFromSnippet) {
        if (snippet.destroyParameters != nullptr) {
          snippet.destroyParameters(m_parameters);
        } else {
          Logger::warn("Leaking the DLSS-NR parameter block: it came from the snippet, which no longer offers DestroyParameters.");
        }
      } else {
        NVSDK_NGX_VULKAN_DestroyParameters(m_parameters);
      }

      m_parameters = nullptr;
    }

    if (m_snippetInitialized && snippet.shutdown1 != nullptr) {
      const NVSDK_NGX_Result result = snippet.shutdown1(m_device->handle());

      if (NVSDK_NGX_FAILED(result)) {
        Logger::warn(str::format("NVSDK_NGX_VULKAN_Shutdown1 failed for DLSS-NR: ", resultToString(result)));
      }

      m_snippetInitialized = false;
    }
  }

  bool NGXNeuralRenderingContext::initializeSnippet() {
    ScopedCpuProfileZone();

    NgxNeuralRenderingSnippet& snippet = NgxNeuralRenderingSnippet::get();

    const std::string exePath = env::getExePath();
    const std::string exeFolder = exePath.substr(0, exePath.find_last_of("\\/"));
    const auto applicationDataPath = str::tows(exeFolder.c_str());

    // Note: the snippet resolves its own weights out of its embedded WEIGHTS_HT resource, so the
    // application data path is only used for its log file.
    NVSDK_NGX_Result result = snippet.initExt(static_cast<unsigned long long>(RtxOptions::applicationId()),
                                              applicationDataPath.c_str(),
                                              m_device->instance()->handle(),
                                              m_device->adapter()->handle(),
                                              m_device->handle(),
                                              NVSDK_NGX_Version_API,
                                              nullptr);

    if (NVSDK_NGX_FAILED(result)) {
      Logger::warn(str::format("NVSDK_NGX_VULKAN_Init_Ext failed for DLSS-NR: ", resultToString(result)));

      if (result == NVSDK_NGX_Result_FAIL_PlatformError) {
        // This is also what the snippet's "Not called from NGX runtime" caller check returns,
        // so point at the one thing that would fix that case.
        Logger::warn("If this snippet build enforces the \"called from NGX runtime\" check, shipping "
                     "remix_nvngx.dll next to the Remix runtime lets the calls through.");
      }

      return false;
    }

    // Note: set before the parameter block is allocated, so that a failure below still tears
    // the snippet down through the destructor's Shutdown1.
    m_snippetInitialized = true;

    // The parameter block is the object every subsequent snippet call operates on, so it comes
    // from the snippet's own AllocateParameters: that keeps the whole flow --- init, parameters,
    // create, evaluate, release, shutdown --- inside nvngx_dlssnr.dll, which is the entire point
    // of loading it by hand. A snippet build that does not export it falls back to the NGX SDK
    // this runtime links against; note that fallback additionally requires the driver's NGX
    // runtime to have been initialized (NGXContext::initialize(), which only runs when a DLSS
    // context is created), so it can fail on machines where DLSS itself is unavailable.
    if (snippet.allocateParameters != nullptr) {
      result = snippet.allocateParameters(&m_parameters);
      m_parametersFromSnippet = true;
    } else {
      result = NVSDK_NGX_VULKAN_AllocateParameters(&m_parameters);
      m_parametersFromSnippet = false;
    }

    if (NVSDK_NGX_FAILED(result) || m_parameters == nullptr) {
      Logger::warn(str::format("AllocateParameters failed for DLSS-NR: ", resultToString(result)));
      m_parameters = nullptr;
      return false;
    }

    return true;
  }

  bool NGXNeuralRenderingContext::initialize(Rc<DxvkContext> renderContext, const uint32_t inputSize[2], const uint32_t outputSize[2]) {
    ScopedCpuProfileZone();

    if (!m_snippetInitialized || m_parameters == nullptr) {
      return false;
    }

    // Keyed on both resolutions: nothing to do when the live feature already matches, and no
    // point retrying a create that already failed at this exact resolution pair.
    const bool sameResolution =
      m_inputSize[0] == inputSize[0] && m_inputSize[1] == inputSize[1] &&
      m_outputSize[0] == outputSize[0] && m_outputSize[1] == outputSize[1];

    if (sameResolution) {
      if (isNeuralRenderingInitialized()) {
        return true;
      }

      if (m_featureCreationFailed) {
        return false;
      }
    }

    if (m_feature) {
      renderContext->getDevice()->waitForIdle();
      releaseNGXFeature();
    }

    m_inputSize[0] = inputSize[0];
    m_inputSize[1] = inputSize[1];
    m_outputSize[0] = outputSize[0];
    m_outputSize[1] = outputSize[1];
    m_featureCreationFailed = false;

    m_parameters->Set(kParamWidth, static_cast<unsigned int>(outputSize[0]));
    m_parameters->Set(kParamHeight, static_cast<unsigned int>(outputSize[1]));
    m_parameters->Set(kParamInputWidth, static_cast<unsigned int>(inputSize[0]));
    m_parameters->Set(kParamInputHeight, static_cast<unsigned int>(inputSize[1]));
    m_parameters->Set(kParamEnabled, static_cast<unsigned int>(1));

    // Only preset 1 exists in this snippet build; anything else logs
    // "preset %d is not available in this DLL build" and loads the same weights anyway.
    m_parameters->Set(kParamRenderPreset, kOnlyPreset);

    // NV-DXVK start: DLSS-NR
    // Note: DLSSNR.ScalingRatio is DEAD in this snippet build. Three separate sites read it and
    // then unconditionally store 1.0f over the result --- uniquely among the float parameters it
    // has no "was it actually set" guard --- so this write can never change anything. It is kept
    // because it is harmless and correct, but it is exactly 1.0 here anyway: this pass does not
    // upscale, so the input and output grids are both the colour grid. Nothing may be predicated
    // on this value.
    const float scalingRatio = inputSize[1] != 0
      ? static_cast<float>(outputSize[1]) / static_cast<float>(inputSize[1])
      : 1.0f;
    m_parameters->Set(kParamScalingRatio, scalingRatio);
    // NV-DXVK end

    // Release video memory when DLSS-NR is disabled.
    m_parameters->Set(NVSDK_NGX_Parameter_FreeMemOnReleaseFeature, 1);

    VkCommandBuffer vkCommandBuffer = renderContext->getCommandList()->getCmdBuffer(dxvk::DxvkCmdBuffer::ExecBuffer);

    NgxNeuralRenderingSnippet& snippet = NgxNeuralRenderingSnippet::get();

    const NVSDK_NGX_Result result = snippet.createFeature(vkCommandBuffer, kFeatureDLSSNR, m_parameters, &m_feature);

    if (NVSDK_NGX_FAILED(result)) {
      Logger::err(str::format("Failed to create DLSS-NR feature: ", resultToString(result)));
      m_feature = nullptr;
      m_initialized = false;
      m_featureCreationFailed = true;
      return false;
    }

    m_initialized = true;

    return true;
  }

  void NGXNeuralRenderingContext::releaseNGXFeature() {
    ScopedCpuProfileZone();

    NgxNeuralRenderingSnippet& snippet = NgxNeuralRenderingSnippet::get();

    if (m_feature && snippet.releaseFeature != nullptr) {
      snippet.releaseFeature(m_feature);
    }

    m_feature = nullptr;

    m_initialized = false;
  }

  bool NGXNeuralRenderingContext::evaluateNeuralRendering(
    Rc<DxvkContext> renderContext,
    const NGXBuffers& buffers,
    const NGXSettings& settings) const {
    if (!m_feature) {
      return false;
    }

    ScopedCpuProfileZone();

    // Color, Depth, MVec and Output are not optional for this feature.
    // NV-DXVK start: DLSS-NR
    // Note: this checks isValid(), not just the pointer --- the extents below dereference
    // ->image directly, and a Resources::Resource can legally exist with a null image/view (for
    // instance a target texture that has not been recreated yet after a resolution change).
    if (buffers.pColor == nullptr || !buffers.pColor->isValid() ||
        buffers.pDepth == nullptr || !buffers.pDepth->isValid() ||
        buffers.pMotionVectors == nullptr || !buffers.pMotionVectors->isValid() ||
        buffers.pOutput == nullptr || !buffers.pOutput->isValid()) {
      return false;
    }
    // NV-DXVK end

    VkCommandBuffer vkCommandbuffer = renderContext->getCommandList()->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);

    // Note: these must stay alive across the EvaluateFeature call --- the snippet reads them
    // through the void pointers set on the parameter block.
    NVSDK_NGX_Resource_VK colorResource = textureToResourceVK(buffers.pColor, false);
    NVSDK_NGX_Resource_VK depthResource = textureToResourceVK(buffers.pDepth, false);
    NVSDK_NGX_Resource_VK motionVectorsResource = textureToResourceVK(buffers.pMotionVectors, false);
    NVSDK_NGX_Resource_VK outputResource = textureToResourceVK(buffers.pOutput, true);
    NVSDK_NGX_Resource_VK controlMaskResource = textureToResourceVK(buffers.pControlMask, false);

    const VkExtent3D colorExtent = buffers.pColor->image->info().extent;
    const VkExtent3D depthExtent = buffers.pDepth->image->info().extent;
    const VkExtent3D motionVectorsExtent = buffers.pMotionVectors->image->info().extent;
    const VkExtent3D outputExtent = buffers.pOutput->image->info().extent;

    static const ResourceParamNames s_colorNames(kParamColor);
    static const ResourceParamNames s_depthNames(kParamDepth);
    static const ResourceParamNames s_motionVectorNames(kParamMVec);
    static const ResourceParamNames s_outputNames(kParamOutput);
    static const ResourceParamNames s_controlMaskNames(kParamControlMask);

    auto setResource = [this](const ResourceParamNames& names, NVSDK_NGX_Resource_VK* resource, uint32_t width, uint32_t height) {
      m_parameters->Set(names.resource.c_str(), static_cast<void*>(resource));
      m_parameters->Set(names.subrectBaseX.c_str(), static_cast<unsigned int>(0));
      m_parameters->Set(names.subrectBaseY.c_str(), static_cast<unsigned int>(0));
      m_parameters->Set(names.subrectWidth.c_str(), static_cast<unsigned int>(width));
      m_parameters->Set(names.subrectHeight.c_str(), static_cast<unsigned int>(height));
    };

    // The parameter block outlives this call --- it is allocated once per context and reused for
    // every frame --- while the NVSDK_NGX_Resource_VK objects above are locals of this frame's
    // call. So an optional resource that is not bound this frame MUST be cleared explicitly:
    // leaving the previous frame's pointer in place both dangles and, for the control mask,
    // keeps the snippet forcing UseAutoMask to 0 (which kills both structure strengths) long
    // after the user turned the mask back off.
    // Note: the explicit void* cast is what picks NVSDK_NGX_Parameter::Set(const char*, void*)
    // over the ID3D11Resource*/ID3D12Resource* overloads.
    auto clearResource = [this](const ResourceParamNames& names) {
      m_parameters->Set(names.resource.c_str(), static_cast<void*>(nullptr));
      m_parameters->Set(names.subrectBaseX.c_str(), static_cast<unsigned int>(0));
      m_parameters->Set(names.subrectBaseY.c_str(), static_cast<unsigned int>(0));
      m_parameters->Set(names.subrectWidth.c_str(), static_cast<unsigned int>(0));
      m_parameters->Set(names.subrectHeight.c_str(), static_cast<unsigned int>(0));
    };

    // NV-DXVK start: DLSS-NR
    // Every resource is described by ITS OWN dimensions, which is what makes colour at output
    // resolution with guides at render resolution work: the snippet validates each rect against
    // that resource's real size and never compares the guide rects against the colour rect. The
    // kernel launch block receives the colour rect, the mvec rect and DLSSNR.MVecScaleX/Y as
    // three independent values.
    //
    // The one cross-resource constraint the snippet does enforce is Color vs Output: with both
    // rects covering their whole resource their dimensions must be equal, or the evaluate is
    // rejected with "Invalid Color/Output rect configuration". DxvkNeuralRendering satisfies
    // that by allocating both the proxy and the neural target at the target (colour) extent.
    setResource(s_colorNames, &colorResource, colorExtent.width, colorExtent.height);
    setResource(s_depthNames, &depthResource, depthExtent.width, depthExtent.height);
    setResource(s_motionVectorNames, &motionVectorsResource, motionVectorsExtent.width, motionVectorsExtent.height);
    setResource(s_outputNames, &outputResource, outputExtent.width, outputExtent.height);
    // NV-DXVK end

    const bool hasControlMask = buffers.pControlMask != nullptr && buffers.pControlMask->isValid();

    if (hasControlMask) {
      const VkExtent3D controlMaskExtent = buffers.pControlMask->image->info().extent;
      setResource(s_controlMaskNames, &controlMaskResource, controlMaskExtent.width, controlMaskExtent.height);
    } else {
      clearResource(s_controlMaskNames);
    }

    m_parameters->Set(kParamEnabled, static_cast<unsigned int>(1));
    m_parameters->Set(kParamReset, static_cast<unsigned int>(settings.resetAccumulation ? 1 : 0));
    m_parameters->Set(kParamDepthInverted, static_cast<unsigned int>(settings.depthInverted ? 1 : 0));
    m_parameters->Set(kParamMVecScaleX, settings.motionVectorScale[0]);
    m_parameters->Set(kParamMVecScaleY, settings.motionVectorScale[1]);

    // Binding an explicit control mask makes the snippet force UseAutoMask to 0 internally,
    // which disables BOTH structure strengths. Only claim the auto mask when nothing is bound.
    m_parameters->Set(kParamUseAutoMask, static_cast<unsigned int>((settings.useAutoMask && !hasControlMask) ? 1 : 0));

    m_parameters->Set(kParamIntensity, settings.intensity);
    m_parameters->Set(kParamLocalToneStrength, settings.localToneStrength);
    m_parameters->Set(kParamLocalStructureStrength, settings.localStructureStrength);
    m_parameters->Set(kParamSkinStructureStrength, settings.skinStructureStrength);
    m_parameters->Set(kParamStyle, static_cast<unsigned int>(settings.style));

    NgxNeuralRenderingSnippet& snippet = NgxNeuralRenderingSnippet::get();

    const NVSDK_NGX_Result result = snippet.evaluateFeature(vkCommandbuffer, m_feature, m_parameters, nullptr);

    if (NVSDK_NGX_FAILED(result)) {
      ONCE(Logger::err(str::format("NVSDK_NGX_VULKAN_EvaluateFeature failed for DLSS-NR: ", resultToString(result))));
      return false;
    }

    return true;
  }
} // namespace dxvk
