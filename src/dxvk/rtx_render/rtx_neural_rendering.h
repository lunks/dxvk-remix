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
   * stands in for them --- and enhances the already resolved image at output resolution.
   *
   * Three stages, because the snippet is a display referred image network and this pass sits
   * upstream of tone mapping where m_finalOutput is still unbounded linear radiance:
   *
   *   1. encode  --- compute, builds a soft clipped sRGB proxy of m_finalOutput
   *   2. evaluate --- the NGX feature, run on the proxy plus the render-res depth/motion guides
   *   3. decode  --- compute, carries the network's change back onto the untouched HDR
   *                  m_finalOutput and restores its original alpha channel
   *
   * Everything downstream (dust particles, bloom, motion blur, tonemapping, post FX, sRGB
   * dither) still sees unbounded linear HDR in m_finalOutput and is unaffected.
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
    // NV-DXVK start: DLSS-NR
    RTX_OPTION("rtx.neuralRendering", bool, requireMatchingGuideResolution, false,
               "Skips the pass whenever the render resolution differs from the output resolution. This is an escape hatch and defaults to disabled.\n"
               "DLSS-NR is fed the already upscaled colour but Remix only has depth and motion vectors at render resolution, so with an upscaler active the guide buffers are on a different grid than the colour.\n"
               "That is a first class configuration for this snippet, not a broken one: it validates every resource against its OWN dimensions, each resource carries its own DLSSNR.<Resource>Subrect* quadruple, and DLSSNR.MVecScaleX/Y converts motion vectors from the guide grid onto the colour grid. Colour at 3840x2160 with guides at 1920x1080 is a known working arrangement.\n"
               "Enable this only to prove that a problem is or is not caused by the mismatched guide grids; with it enabled the pass does nothing at all whenever an upscaler is active.");

    // The HDR colour codec. This pass runs before tone mapping (rtx_context.cpp:748 vs 758), so
    // m_finalOutput is unbounded linear path traced radiance, while DLSS-NR is a display
    // referred image network. The codec builds a soft clipped, sRGB encoded proxy for the
    // network and carries its answer back onto the untouched HDR original; see
    // neural_rendering_decode.comp.slang for the derivation.
    RTX_OPTION_ARGS("rtx.neuralRendering", float, paperWhiteScale, 1.0f,
                    "Divides the post exposure colour before the proxy image's soft clip knee, i.e. the value that ends up treated as display white.\n"
                    "Remix has no paper white of its own --- there is no HDR swapchain path in the runtime --- so by default the scene linear to display referred conversion is taken from the tonemapper's own exposure (see trackAutoExposure) and this is only a trim on top of it, which is why the default is 1.0 rather than the 16.0 a RenoDX style fixed scale would use.\n"
                    "Raise it if the proxy looks blown out (highlights crushed into the soft clip shoulder), lower it if the proxy looks black.",
                    args.minValue = 0.01f, args.maxValue = 64.0f);
    RTX_OPTION("rtx.neuralRendering", bool, trackAutoExposure, true,
               "Folds the tonemapper's auto exposure value into the proxy scale, so the soft clip knee tracks scene brightness instead of sitting at a fixed radiance.\n"
               "Disable this to get a fixed scale, which is what the RenoDX deployment does; the RenoDX equivalent of NRPaperWhiteScale=16 is this disabled and paperWhiteScale set to 16.\n"
               "Note the exposure value read at this point in the frame is one frame old, because DxvkAutoExposure only runs later, from dispatchToneMapping.");
    RTX_OPTION_ARGS("rtx.neuralRendering", float, transferStrength, 1.0f,
                    "How much of the network's change is carried back onto the HDR original. 0.0 is an exact bypass: the frame is returned bit for bit unchanged.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION_ARGS("rtx.neuralRendering", float, colorStrength, 1.0f,
                    "0.0 keeps the original's chromaticity exactly and transfers only the network's luminance change; 1.0 takes the network's colour as well.\n"
                    "Lower this if the image picks up a colour cast.\n"
                    "Below a display referred luminance of about 1/1000 of diffuse white the chromaticity preserving path is faded back out to the network's own colour, because there the original's hue is path tracer noise and renormalising it to the network's luminance would turn it into a saturated speckle.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    // NV-DXVK end

  protected:
    virtual bool isEnabled() const override;

    // Releases the NGX feature, the parameter block and the snippet's per-device init state
    // when the pass is turned off. RtxPass only frees the target texture for us.
    virtual void onDeactivation() override;

    virtual void createTargetResource(Rc<DxvkContext>& ctx, const VkExtent3D& targetExtent) override;
    virtual void releaseTargetResource() override;

  private:
    // NV-DXVK start: DLSS-NR
    // Static half of the scene linear -> display referred scale used to build the proxy. The
    // scene adaptive half comes from the auto exposure texture inside the shaders, so encode
    // and decode agree by construction; this value is computed once per frame and handed to
    // both so they cannot disagree either.
    static float calcProxyScale();

    // Runs the HDR colour codec's encode half: m_finalOutput -> m_neuralRenderingProxy.
    void dispatchProxyEncode(
      Rc<RtxContext> ctx,
      const Resources::RaytracingOutput& rtOutput,
      const Rc<DxvkImageView>& exposureView,
      bool autoExposureEnabled,
      float proxyScale,
      const VkExtent3D& colorExtent);

    // Runs the HDR colour codec's decode half, folding the neural result back into
    // m_finalOutput while preserving its alpha channel.
    void dispatchProxyDecode(
      Rc<RtxContext> ctx,
      const Resources::RaytracingOutput& rtOutput,
      const Rc<DxvkImageView>& exposureView,
      bool autoExposureEnabled,
      float proxyScale,
      const VkExtent3D& colorExtent);
    // NV-DXVK end

    Resources::Resource m_neuralRenderingOutput;
    // NV-DXVK start: DLSS-NR
    // The display referred image DLSS-NR actually sees. Lives at the colour (target) extent
    // because the snippet rejects an evaluate whose Color and Output rects differ in size.
    Resources::Resource m_neuralRenderingProxy;
    // NV-DXVK end

    std::unique_ptr<NGXNeuralRenderingContext> m_neuralRenderingContext;
    bool m_contextCreationFailed = false;
    bool m_loggedGuideResolutionMismatch = false;
    // NV-DXVK start: DLSS-NR
    // The guide grid the live temporal history was accumulated against. The NGX feature is
    // keyed on the colour grid only, so changing DLSS quality at a fixed output resolution
    // moves the motion vector grid (and DLSSNR.MVecScaleX/Y with it) underneath an unchanged
    // history. Latch it and force a one frame reset when it moves.
    uint32_t m_guideExtent[2] = {};
    // NV-DXVK end
  };
} // namespace dxvk
