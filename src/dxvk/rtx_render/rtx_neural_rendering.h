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

#include <memory>

#include "../dxvk_include.h"
#include "rtx_resources.h"
#include "rtx_ngx_neural_rendering.h"

namespace dxvk {
  class DxvkCommandList;
  class DxvkBarrierSet;
  class DxvkContext;
  class RtxContext;

  /**
   * DLSS Neural Rendering (DLSS-NR) post pass.
   *
   * Runs after the upscaler --- after DLSS, DLSS-RR, XeSS, NIS, TAA-U or the plain copy that
   * stands in for them --- and enhances the already resolved image at output resolution. It
   * reads m_finalOutput, writes its own target sized texture and copies the result back into
   * m_finalOutput so every downstream pass (dust particles, bloom, motion blur, tonemapping,
   * post FX, sRGB dither) is unaffected.
   *
   * The pass is modelled on DxvkRayReconstruction (rtx_ray_reconstruction.h:29) --- same
   * CommonDeviceObject + RtxPass base, same barrier shape around the NGX evaluate, same lazy
   * NGX context creation --- but it is not an upscaler and never participates in
   * RtxContext::getCurrentFrameUpscaler().
   *
   * It silently does nothing when nvngx_dlssnr.dll is not shipped beside the module.
   */
  class DxvkNeuralRendering : public CommonDeviceObject, public RtxPass {
  public:
    explicit DxvkNeuralRendering(DxvkDevice* device);
    ~DxvkNeuralRendering();

    DxvkNeuralRendering(const DxvkNeuralRendering&)                = delete;
    DxvkNeuralRendering(DxvkNeuralRendering&&) noexcept            = delete;
    DxvkNeuralRendering& operator=(const DxvkNeuralRendering&)     = delete;
    DxvkNeuralRendering& operator=(DxvkNeuralRendering&&) noexcept = delete;

    /** True when nvngx_dlssnr.dll was found and could be driven. */
    bool supportsNeuralRendering() const;

    /** Reason supportsNeuralRendering() is false, for the UI and the log. */
    const std::string& getNotSupportedReason() const;

    void dispatch(
      Rc<RtxContext> ctx,
      DxvkBarrierSet& barriers,
      const Resources::RaytracingOutput& rtOutput,
      bool resetHistory = false);

    void showImguiSettings();

    void release();

    void onDestroy();

    RTX_OPTION_ARGS("rtx.neuralRendering", bool, enable, false,
                    "Enables DLSS Neural Rendering, a post process neural enhancement applied to the resolved image after upscaling.\n"
                    "Requires nvngx_dlssnr.dll to be present next to the Remix runtime; the pass does nothing when it is not.",
                    args.environment = "RTX_NEURAL_RENDERING_ENABLE",
                    args.flags = RtxOptionFlags::UserSetting);

    // Note on the five tuning values below: the defaults are the snippet's own fallbacks, read
    // out of its parameter block. The scale they sit on is not known --- 1.0 is what the DLL
    // substitutes when the host supplies nothing, it is not a verified neutral midpoint.
    RTX_OPTION_ARGS("rtx.neuralRendering", float, intensity, 1.0f,
                    "Overall strength of the Neural Rendering effect (DLSSNR.Intensity). 1.0 is the snippet's own fallback value.",
                    args.minValue = 0.0f);
    RTX_OPTION_ARGS("rtx.neuralRendering", float, localToneStrength, 1.0f,
                    "Strength of the local tone term (DLSSNR.LocalToneStrength). 1.0 is the snippet's own fallback value.",
                    args.minValue = 0.0f);
    RTX_OPTION_ARGS("rtx.neuralRendering", float, localStructureStrength, 1.0f,
                    "Strength of the local structure term (DLSSNR.LocalStructureStrength). 1.0 is the snippet's own fallback value.\n"
                    "Has no effect unless useAutoMask is enabled and no control mask is bound: with the auto mask off the snippet forces this to -1 internally.",
                    args.minValue = 0.0f);
    RTX_OPTION("rtx.neuralRendering", float, skinStructureStrength, -1.0f,
               "Strength of the skin structure term (DLSSNR.SkinStructureStrength).\n"
               "Any negative value is an explicit sentinel meaning \"inherit localStructureStrength\", which is the default. 0.0 is not neutral, it flattens skin structure.\n"
               "Has no effect unless useAutoMask is enabled and no control mask is bound.");
    RTX_OPTION("rtx.neuralRendering", uint, style, 0,
               "Style index passed to the snippet (DLSSNR.Style). 0 is the snippet's own fallback value.");

    RTX_OPTION("rtx.neuralRendering", bool, useAutoMask, true,
               "Lets the snippet derive its own control mask (DLSSNR.UseAutoMask).\n"
               "This gates BOTH structure strengths: with it disabled the snippet forces localStructureStrength and skinStructureStrength to -1 and neither does anything.\n"
               "It is forced off whenever useControlMask is enabled, because binding an explicit control mask makes the snippet clear it internally.");
    RTX_OPTION("rtx.neuralRendering", bool, useControlMask, false,
               "Binds the shared bias current colour mask as the explicit DLSS-NR control mask (DLSSNR.ControlMask).\n"
               "Enabling this disables the snippet's auto mask and therefore both structure strength controls.");
    RTX_OPTION("rtx.neuralRendering", bool, requireMatchingGuideResolution, true,
               "Skips the pass whenever the render resolution differs from the output resolution.\n"
               "DLSS-NR is fed the already upscaled colour but Remix only has depth and motion vectors at render resolution, so with an upscaler active the guide buffers are on a different grid than the colour.\n"
               "Disable this to run the pass anyway, in which case the guides are described to the snippet only by their own subrect sizes and by DLSSNR.MVecScaleX/Y. That combination is untested.");

  protected:
    virtual bool isEnabled() const override;

    // Releases the NGX feature, the parameter block and the snippet's per-device init state
    // when the pass is turned off. RtxPass only frees the target texture for us.
    virtual void onDeactivation() override;

    virtual void createTargetResource(Rc<DxvkContext>& ctx, const VkExtent3D& targetExtent) override;
    virtual void releaseTargetResource() override;

  private:
    Resources::Resource m_neuralRenderingOutput;

    std::unique_ptr<NGXNeuralRenderingContext> m_neuralRenderingContext;
    bool m_contextCreationFailed = false;
    bool m_loggedGuideResolutionMismatch = false;
  };
} // namespace dxvk
