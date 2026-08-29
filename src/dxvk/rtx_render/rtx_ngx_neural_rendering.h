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
#pragma once

// Note: this gets included from other modules (dxvk_objects.h), so use the full path to
// external the same way rtx_ngx_wrapper.h and rtx_dlss.h do --- ugly!
#ifdef _M_X64
#include "../../../external/ngx_sdk_dldn/include/nvsdk_ngx.h"
#else
#include "../../../external/ngx_sdk_dldn_arm64/include/nvsdk_ngx.h"
#endif

#include <memory>
#include <string>

#include "../util/rc/util_rc_ptr.h"
#include "rtx_resources.h"

namespace dxvk {
  class DxvkDevice;
  class DxvkContext;

  /**
   * NGX context for DLSS Neural Rendering (DLSS-NR), NGX feature id 18, snippet
   * nvngx_dlssnr.dll.
   *
   * Shaped after NGXRayReconstructionContext (rtx_ngx_wrapper.h:225) but deliberately NOT
   * derived from NGXFeatureContext, because it does not go through the driver's nvngx.dll:
   *
   *   * nvngx.dll verifies the Authenticode signature of every snippet it loads and enforces
   *     the snippet's NGXMinimumDriverVersion (615.00) and NGXGpuArchitecture (Blackwell2)
   *     resource strings. A snippet patched to run on Ada fails all three, so a driver-routed
   *     load can never succeed for us.
   *   * The snippet exports the whole NGX Vulkan surface itself, so it can be driven
   *     standalone once the snippet module is loaded by hand.
   *
   * See NgxNeuralRenderingSnippet in rtx_ngx_neural_rendering.cpp for the loading rules,
   * including the optional remix_nvngx.dll call trampoline and the snippet-side "called from
   * NGX runtime" caller check it exists to satisfy.
   *
   * The parameter block comes from the snippet's own NVSDK_NGX_VULKAN_AllocateParameters, so
   * init, parameters, create, evaluate, release and shutdown all stay inside the snippet. A
   * build that does not export it falls back to the linked NGX SDK's
   * NVSDK_NGX_VULKAN_AllocateParameters, which does require the driver's NGX runtime to have
   * been initialized elsewhere.
   */
  class NGXNeuralRenderingContext final {
  public:
    /**
     * Resources handed to the snippet. Color, Depth, MVec and Output are required by the
     * feature; ControlMask is optional and may be nullptr.
     */
    struct NGXBuffers {
      const Resources::Resource* pColor;
      const Resources::Resource* pDepth;
      const Resources::Resource* pMotionVectors;
      const Resources::Resource* pOutput;
      const Resources::Resource* pControlMask;
    };

    /**
     * Per-evaluate state. The tuning defaults here are the snippet's OWN fallbacks, read out
     * of its parameter block: each read is followed by a `cmp eax,0xbad00000` test that
     * substitutes the value below when the host supplied nothing.
     *
     * Note: the scale these tuning values sit on is not known. 1.0 is the snippet's fallback,
     * not a verified neutral midpoint --- do not describe it as "neutral".
     */
    struct NGXSettings {
      bool resetAccumulation = false;
      bool depthInverted = false;
      float motionVectorScale[2] = { 1.0f, 1.0f };

      float intensity = 1.0f;
      float localToneStrength = 1.0f;
      float localStructureStrength = 1.0f;
      // Negative means "inherit localStructureStrength": the snippet does an explicit comiss
      // against 0 and a jae, and copies the local structure strength on the other branch.
      // 0.0f is NOT neutral, it flattens skin structure.
      float skinStructureStrength = -1.0f;
      uint32_t style = 0;
      // Gates BOTH structure strengths. With this at 0 the snippet internally forces
      // localStructureStrength and skinStructureStrength to -1 and neither does anything.
      // Binding an explicit ControlMask also forces this to 0 inside the snippet, so
      // evaluateNeuralRendering() only ever sets it when no ControlMask is bound.
      bool useAutoMask = true;
    };

    /**
     * Creates a context, or returns nullptr when DLSS-NR cannot run in this process.
     * Returning nullptr is the normal, non-fatal outcome when nvngx_dlssnr.dll is not shipped
     * beside the module --- the caller must treat it as "feature absent", never as an error.
     */
    static std::unique_ptr<NGXNeuralRenderingContext> createNeuralRenderingContext(DxvkDevice* device);

    /**
     * True when nvngx_dlssnr.dll could be loaded and all required exports resolved. Safe to
     * call before any context exists; the underlying load happens exactly once per process.
     */
    static bool isSnippetAvailable();

    /**
     * Human readable reason for isSnippetAvailable() being false, for the UI and the log.
     * Empty when the snippet is available.
     */
    static const std::string& getSnippetNotAvailableReason();

    ~NGXNeuralRenderingContext();

    NGXNeuralRenderingContext(const NGXNeuralRenderingContext&)                = delete;
    NGXNeuralRenderingContext(NGXNeuralRenderingContext&&) noexcept            = delete;
    NGXNeuralRenderingContext& operator=(const NGXNeuralRenderingContext&)     = delete;
    NGXNeuralRenderingContext& operator=(NGXNeuralRenderingContext&&) noexcept = delete;

    /**
     * Creates (or re-creates) the DLSS-NR feature for a given input/output resolution pair.
     * Keyed on both resolutions: a call that matches the resolutions the live feature was
     * built with is a no-op and returns true, so this is safe to call every frame.
     * Returns false when the feature could not be created; the caller must then skip the pass.
     */
    bool initialize(Rc<DxvkContext> renderContext, const uint32_t inputSize[2], const uint32_t outputSize[2]);

    /** Release DLSS-NR. */
    void releaseNGXFeature();

    /** Checks if DLSS-NR is initialized. */
    bool isNeuralRenderingInitialized() const {
      return m_initialized && m_feature != nullptr;
    }

    /** Evaluate DLSS-NR. */
    bool evaluateNeuralRendering(Rc<DxvkContext> renderContext, const NGXBuffers& buffers, const NGXSettings& settings) const;

  public:
    // note: ctor is public due to make_unique/unique_ptr, but not intended as public --- use
    // NGXNeuralRenderingContext::createNeuralRenderingContext instead
    explicit NGXNeuralRenderingContext(DxvkDevice* device);

  private:
    bool initializeSnippet();

    DxvkDevice* m_device = nullptr;
    NVSDK_NGX_Parameter* m_parameters = nullptr;
    // Which side allocated m_parameters, so it is released through the same one.
    bool m_parametersFromSnippet = false;
    NVSDK_NGX_Handle* m_feature = nullptr;

    bool m_initialized = false;
    bool m_snippetInitialized = false;
    // Latched so a feature that failed to create is not retried on every single frame. Cleared
    // whenever the resolution pair changes, so a resize is still allowed to try again.
    bool m_featureCreationFailed = false;

    uint32_t m_inputSize[2] = {};
    uint32_t m_outputSize[2] = {};
  };
} // namespace dxvk
