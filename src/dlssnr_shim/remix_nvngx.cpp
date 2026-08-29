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

// remix_nvngx.dll --- call trampoline for the DLSS-NR NGX snippet (nvngx_dlssnr.dll).
//
// Why this module exists
// ----------------------
// Every usable export of nvngx_dlssnr.dll (DLSSNR 310.8.0) opens with, in effect:
//
//     GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
//                        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
//                        <return address>, &callerModule);
//     GetModuleFileNameW(callerModule, path, MAX_PATH);
//     if (!wcsstr(path, L"nvngx.dll")) {
//       log("Error: Not called from NGX runtime - %S", path);
//       return NVSDK_NGX_Result_FAIL_PlatformError;   // 0xbad00002
//     }
//
// The check is on the CALLING module's file path, and it is a plain case sensitive wide
// substring search.
//
// This module's file name contains "nvngx.dll" as a substring, so a call issued from here
// satisfies such a check. It holds no state beyond the resolved snippet entry points, does no
// loading of its own, and is only ever driven by NGXNeuralRenderingContext.
//
// It is REQUIRED in practice. No direct call into a gated export has ever been observed to
// succeed without defeating the check: Init_Ext and CreateFeature are both gated, so without
// this module initialisation fails with 0xbad00002 and DLSS-NR reports itself unsupported.
//
// The one known-working direct-load configuration (RTX 4090, driver 610.43.02) is RenoDX's
// ReShade add-on on D3D12 -- and it works only because it patches the snippet's own
// GetModuleFileNameW import so the check reads a satisfying path. Its error string
// "signed feature has no GetModuleFileNameW import" is that workaround failing. That is
// evidence the gate is enforced, not evidence that direct calls work.
//
// NgxNeuralRenderingSnippet::load() still falls back to calling the snippet directly when this
// module is absent. That fallback exists to fail loudly with a clear remedy in the log, not
// because it is expected to work: a resolve-time probe cannot detect the gate, because
// GetProcAddress succeeds and only the calls fail.
//
// Gated exports:   CreateFeature, CreateFeature1, GetFeatureRequirements,
//                  GetScratchBufferSize, Init_Ext, Init_Ext2, PopulateParameters_Impl,
//                  ReleaseFeature, Shutdown, Shutdown1.
// Ungated exports: EvaluateFeature, GetFeatureDeviceExtensionRequirements,
//                  GetFeatureInstanceExtensionRequirements.
// EvaluateFeature is forwarded through here anyway so that the whole feature lifetime is
// issued from one module.
//
// The trampoline deliberately uses opaque pointer types rather than the NGX headers: it must
// not acquire a dependency on the NGX SDK drop, and every parameter it forwards is either a
// 64 bit handle or a 32 bit enum, so the ABI is identical either way.

#include <windows.h>

namespace {
  // NVSDK_NGX_Result, NVSDK_NGX_Version and NVSDK_NGX_Feature are all plain C enums in the NGX
  // headers, so they are passed as 32 bit values.
  typedef unsigned int NgxResult;
  typedef unsigned int NgxVersion;
  typedef unsigned int NgxFeature;

  // NVSDK_NGX_Result_FAIL_NotInitialized
  constexpr NgxResult kNgxResultFailNotInitialized = 0xbad00007u;

  typedef NgxResult(__cdecl* PfnInitExt)(unsigned long long appId, const wchar_t* applicationDataPath,
                                         void* instance, void* physicalDevice, void* device,
                                         NgxVersion sdkVersion, const void* featureCommonInfo);
  typedef NgxResult(__cdecl* PfnShutdown1)(void* device);
  typedef NgxResult(__cdecl* PfnCreateFeature)(void* commandBuffer, NgxFeature featureId, void* parameters, void** outHandle);
  typedef NgxResult(__cdecl* PfnReleaseFeature)(void* handle);
  typedef NgxResult(__cdecl* PfnEvaluateFeature)(void* commandBuffer, const void* handle, const void* parameters, void* progressCallback);
  typedef NgxResult(__cdecl* PfnAllocateParameters)(void** outParameters);
  typedef NgxResult(__cdecl* PfnDestroyParameters)(void* parameters);

  PfnInitExt g_initExt = nullptr;
  PfnShutdown1 g_shutdown1 = nullptr;
  PfnCreateFeature g_createFeature = nullptr;
  PfnReleaseFeature g_releaseFeature = nullptr;
  PfnEvaluateFeature g_evaluateFeature = nullptr;
  PfnAllocateParameters g_allocateParameters = nullptr;
  PfnDestroyParameters g_destroyParameters = nullptr;

  // Written after every forwarded call. The only purpose of this store is to keep the compiler
  // from turning `return g_xxx(...)` into a tail call: a tail call reuses the caller's return
  // address, which would put the snippet's caller check back on d3d9.dll and defeat the entire
  // point of this module.
  volatile long g_forwardedCallCount = 0;
}

extern "C" {

  // Hands the trampoline the already loaded nvngx_dlssnr.dll. Passing nullptr clears the
  // resolved entry points.
  __declspec(dllexport) void __cdecl RemixNgxTrampoline_SetSnippet(void* snippetModule) {
    const HMODULE module = reinterpret_cast<HMODULE>(snippetModule);

    if (module == nullptr) {
      g_initExt = nullptr;
      g_shutdown1 = nullptr;
      g_createFeature = nullptr;
      g_releaseFeature = nullptr;
      g_evaluateFeature = nullptr;
      g_allocateParameters = nullptr;
      g_destroyParameters = nullptr;
      return;
    }

    g_initExt = reinterpret_cast<PfnInitExt>(GetProcAddress(module, "NVSDK_NGX_VULKAN_Init_Ext"));
    g_shutdown1 = reinterpret_cast<PfnShutdown1>(GetProcAddress(module, "NVSDK_NGX_VULKAN_Shutdown1"));
    g_createFeature = reinterpret_cast<PfnCreateFeature>(GetProcAddress(module, "NVSDK_NGX_VULKAN_CreateFeature"));
    g_releaseFeature = reinterpret_cast<PfnReleaseFeature>(GetProcAddress(module, "NVSDK_NGX_VULKAN_ReleaseFeature"));
    g_evaluateFeature = reinterpret_cast<PfnEvaluateFeature>(GetProcAddress(module, "NVSDK_NGX_VULKAN_EvaluateFeature"));
    // Note: these two may not exist on every snippet build, in which case the forwarders below
    // return kNgxResultFailNotInitialized. The caller never gets that far --- it inspects the
    // snippet's own export table before deciding to use a parameter entry point at all, and
    // falls back to the NGX SDK it links against when the snippet has none.
    g_allocateParameters = reinterpret_cast<PfnAllocateParameters>(GetProcAddress(module, "NVSDK_NGX_VULKAN_AllocateParameters"));
    g_destroyParameters = reinterpret_cast<PfnDestroyParameters>(GetProcAddress(module, "NVSDK_NGX_VULKAN_DestroyParameters"));
  }

  __declspec(dllexport) NgxResult __cdecl NVSDK_NGX_VULKAN_Init_Ext(
      unsigned long long appId,
      const wchar_t* applicationDataPath,
      void* instance,
      void* physicalDevice,
      void* device,
      NgxVersion sdkVersion,
      const void* featureCommonInfo) {
    if (g_initExt == nullptr) {
      return kNgxResultFailNotInitialized;
    }

    const NgxResult result = g_initExt(appId, applicationDataPath, instance, physicalDevice, device, sdkVersion, featureCommonInfo);
    ++g_forwardedCallCount;

    return result;
  }

  __declspec(dllexport) NgxResult __cdecl NVSDK_NGX_VULKAN_Shutdown1(void* device) {
    if (g_shutdown1 == nullptr) {
      return kNgxResultFailNotInitialized;
    }

    const NgxResult result = g_shutdown1(device);
    ++g_forwardedCallCount;

    return result;
  }

  __declspec(dllexport) NgxResult __cdecl NVSDK_NGX_VULKAN_CreateFeature(
      void* commandBuffer,
      NgxFeature featureId,
      void* parameters,
      void** outHandle) {
    if (g_createFeature == nullptr) {
      return kNgxResultFailNotInitialized;
    }

    const NgxResult result = g_createFeature(commandBuffer, featureId, parameters, outHandle);
    ++g_forwardedCallCount;

    return result;
  }

  __declspec(dllexport) NgxResult __cdecl NVSDK_NGX_VULKAN_ReleaseFeature(void* handle) {
    if (g_releaseFeature == nullptr) {
      return kNgxResultFailNotInitialized;
    }

    const NgxResult result = g_releaseFeature(handle);
    ++g_forwardedCallCount;

    return result;
  }

  __declspec(dllexport) NgxResult __cdecl NVSDK_NGX_VULKAN_AllocateParameters(void** outParameters) {
    if (g_allocateParameters == nullptr) {
      return kNgxResultFailNotInitialized;
    }

    const NgxResult result = g_allocateParameters(outParameters);
    ++g_forwardedCallCount;

    return result;
  }

  __declspec(dllexport) NgxResult __cdecl NVSDK_NGX_VULKAN_DestroyParameters(void* parameters) {
    if (g_destroyParameters == nullptr) {
      return kNgxResultFailNotInitialized;
    }

    const NgxResult result = g_destroyParameters(parameters);
    ++g_forwardedCallCount;

    return result;
  }

  __declspec(dllexport) NgxResult __cdecl NVSDK_NGX_VULKAN_EvaluateFeature(
      void* commandBuffer,
      const void* handle,
      const void* parameters,
      void* progressCallback) {
    if (g_evaluateFeature == nullptr) {
      return kNgxResultFailNotInitialized;
    }

    const NgxResult result = g_evaluateFeature(commandBuffer, handle, parameters, progressCallback);
    ++g_forwardedCallCount;

    return result;
  }
}
