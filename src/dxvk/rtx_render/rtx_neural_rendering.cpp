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
#include <algorithm>
#include <cmath>
#include <vector>

#include "rtx_neural_rendering.h"

#include "rtx.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_imgui.h"
#include "rtx_ngx_neural_rendering.h"
#include "rtx_ray_reconstruction.h"
// NV-DXVK start: DLSS-NR
#include "rtx_auto_exposure.h"
#include "rtx_render/rtx_shader_manager.h"
#include "../util/util_once.h"
#include "rtx/pass/neural_rendering/neural_rendering.h"

#include <rtx_shaders/neural_rendering_encode.h>
#include <rtx_shaders/neural_rendering_decode.h>
// NV-DXVK end

#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"

namespace dxvk {

  // NV-DXVK start: DLSS-NR
  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    class NeuralRenderingEncodeShader : public ManagedShader {
      SHADER_SOURCE(NeuralRenderingEncodeShader, VK_SHADER_STAGE_COMPUTE_BIT, neural_rendering_encode)

      PUSH_CONSTANTS(NeuralRenderingEncodeArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE2D_READONLY(NEURAL_RENDERING_ENCODE_COLOR_INPUT)
        RW_TEXTURE1D_READONLY(NEURAL_RENDERING_ENCODE_EXPOSURE_INPUT)
        RW_TEXTURE2D(NEURAL_RENDERING_ENCODE_PROXY_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(NeuralRenderingEncodeShader);

    class NeuralRenderingDecodeShader : public ManagedShader {
      SHADER_SOURCE(NeuralRenderingDecodeShader, VK_SHADER_STAGE_COMPUTE_BIT, neural_rendering_decode)

      PUSH_CONSTANTS(NeuralRenderingDecodeArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE2D_READONLY(NEURAL_RENDERING_DECODE_PROXY_INPUT)
        RW_TEXTURE2D_READONLY(NEURAL_RENDERING_DECODE_NEURAL_INPUT)
        RW_TEXTURE1D_READONLY(NEURAL_RENDERING_DECODE_EXPOSURE_INPUT)
        RW_TEXTURE2D(NEURAL_RENDERING_DECODE_COLOR_INPUT_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(NeuralRenderingDecodeShader);
  }
  // NV-DXVK end

  DxvkNeuralRendering::DxvkNeuralRendering(DxvkDevice* device)
    : CommonDeviceObject(device)
    , RtxPass(device) {
  }

  DxvkNeuralRendering::~DxvkNeuralRendering() {
    release();
  }

  bool DxvkNeuralRendering::supportsNeuralRendering() const {
    return NGXNeuralRenderingContext::isSnippetAvailable();
  }

  const std::string& DxvkNeuralRendering::getNotSupportedReason() const {
    return NGXNeuralRenderingContext::getSnippetNotAvailableReason();
  }

  bool DxvkNeuralRendering::isEnabled() const {
    // Note: the short circuit matters --- isSnippetAvailable() performs the one time
    // LoadLibraryW of nvngx_dlssnr.dll, and there is no reason to pay for it, or to log about
    // a missing snippet, on installs that never turn the feature on.
    return enable() && NGXNeuralRenderingContext::isSnippetAvailable();
  }

  void DxvkNeuralRendering::createTargetResource(Rc<DxvkContext>& ctx, const VkExtent3D& targetExtent) {
    // Note: matches the format of Resources::RaytracingOutput::m_finalOutput
    // (rtx_resources.cpp:1254), which is both the source and the final destination of this pass.
    m_neuralRenderingOutput = Resources::createImageResource(ctx, "neural rendering output", targetExtent, VK_FORMAT_R16G16B16A16_SFLOAT);
    // NV-DXVK start: DLSS-NR
    // The proxy is what the snippet is handed as DLSSNR.Color, so it has to be at the colour
    // (target) extent: the snippet validates the Color rect against the Output rect and
    // rejects the evaluate outright when their dimensions differ. FP16 is what the RenoDX
    // deployment uses for the same surface ("FP16 working surface"), and it comfortably holds
    // the [0, 1] sRGB encoded values the network expects.
    m_neuralRenderingProxy = Resources::createImageResource(ctx, "neural rendering proxy", targetExtent, VK_FORMAT_R16G16B16A16_SFLOAT);
    // NV-DXVK end
  }

  void DxvkNeuralRendering::releaseTargetResource() {
    m_neuralRenderingOutput.reset();
    // NV-DXVK start: DLSS-NR
    m_neuralRenderingProxy.reset();
    // NV-DXVK end
  }

  void DxvkNeuralRendering::release() {
    m_neuralRenderingContext = nullptr;
    m_contextCreationFailed = false;
    m_loggedGuideResolutionMismatch = false;
    // NV-DXVK start: DLSS-NR
    m_guideExtent[0] = 0;
    m_guideExtent[1] = 0;
    // NV-DXVK end
  }

  void DxvkNeuralRendering::onDestroy() {
    release();
  }

  void DxvkNeuralRendering::onDeactivation() {
    // Turning the pass off has to give the NGX feature (and the video memory
    // NVSDK_NGX_Parameter_FreeMemOnReleaseFeature exists to reclaim), the parameter block and
    // the snippet's Init_Ext state back, not just the target texture RtxPass releases for us.
    // The wait matches how the runtime releases an NGX context when the upscaler changes
    // (rtx_context.cpp:422-433): the feature may still be referenced by in-flight work.
    if (m_neuralRenderingContext != nullptr) {
      m_device->waitForIdle();
      release();
    }
  }

  void DxvkNeuralRendering::dispatch(
      Rc<RtxContext> ctx,
      DxvkBarrierSet& barriers,
      const Resources::RaytracingOutput& rtOutput,
      bool resetHistory) {
    ScopedGpuProfileZone(ctx, "Neural Rendering");
    ctx->setFramePassStage(RtxFramePassStage::NeuralRendering);

    // NV-DXVK start: DLSS-NR
    // Note: each condition is reported separately and once. A silent return here is
    // indistinguishable from the pass working perfectly, which has already cost one round of
    // hardware testing --- the frame looks correct either way.
    if (!isActive()) {
      ONCE(Logger::info("NVIDIA DLSS-NR inactive: the pass is not enabled for this frame."));
      return;
    }

    if (!m_neuralRenderingOutput.isValid() || !m_neuralRenderingProxy.isValid()) {
      ONCE(Logger::warn(str::format(
        "NVIDIA DLSS-NR skipped: pass targets are not allocated (output valid: ",
        m_neuralRenderingOutput.isValid() ? "yes" : "no",
        ", proxy valid: ", m_neuralRenderingProxy.isValid() ? "yes" : "no",
        "). createTargetResource() has not run for this resolution.")));
      return;
    }
    // NV-DXVK end

    if (m_neuralRenderingContext == nullptr) {
      if (m_contextCreationFailed) {
        return;
      }

      m_neuralRenderingContext = NGXNeuralRenderingContext::createNeuralRenderingContext(m_device);

      if (m_neuralRenderingContext == nullptr) {
        // Note: createNeuralRenderingContext() has already logged why. Latch the failure so the
        // renderer is not asked to retry every frame, and carry on without the pass.
        m_contextCreationFailed = true;
        return;
      }
    }

    // The colour handed to DLSS-NR has already been resolved by the upscaler, so it is at target
    // extent, while Remix only produces depth and motion vectors at render extent.
    const VkExtent3D colorExtent = rtOutput.m_finalOutputExtent;
    const VkExtent3D guideExtent = rtOutput.m_compositeOutputExtent;

    // NV-DXVK start: DLSS-NR
    // Note: this is an escape hatch and defaults off. Guides on a different grid than the
    // colour is a first class configuration for this snippet --- it validates each resource
    // against that resource's own dimensions, every resource carries its own
    // DLSSNR.<Resource>Subrect* quadruple, and DLSSNR.MVecScaleX/Y converts the motion vectors
    // onto the colour grid. Colour at 4K with guides at 1080p is a known working arrangement.
    // With an upscaler active this branch means the pass does nothing at all, which is exactly
    // what the first on-hardware run showed.
    if (requireMatchingGuideResolution() &&
        (colorExtent.width != guideExtent.width || colorExtent.height != guideExtent.height)) {
      if (!m_loggedGuideResolutionMismatch) {
        Logger::warn(str::format("NVIDIA DLSS-NR skipped: render resolution ", guideExtent.width, "x", guideExtent.height,
                                 " does not match output resolution ", colorExtent.width, "x", colorExtent.height,
                                 ", and rtx.neuralRendering.requireMatchingGuideResolution is enabled. "
                                 "Set it back to False (the default) to run the pass."));
        m_loggedGuideResolutionMismatch = true;
      }

      return;
    }

    // The NGX feature is keyed on the colour grid alone, so switching DLSS quality at a fixed
    // output resolution moves the guide grid --- and DLSSNR.MVecScaleX/Y with it --- underneath
    // a temporal history that was accumulated against the old one. Nothing else notices, so
    // latch the guide extent here and force a single reset frame when it moves.
    bool resetGuideHistory = false;

    if (m_guideExtent[0] != guideExtent.width || m_guideExtent[1] != guideExtent.height) {
      resetGuideHistory = m_guideExtent[0] != 0 || m_guideExtent[1] != 0;
      m_guideExtent[0] = guideExtent.width;
      m_guideExtent[1] = guideExtent.height;
      // Let the mismatch warning fire again if the option is turned back on after a resize.
      m_loggedGuideResolutionMismatch = false;
    }
    // NV-DXVK end

    // DLSSNR.Width/DLSSNR.Height describe the resource bound as DLSSNR.Color, which is the
    // proxy image at output resolution --- this pass does not upscale, so the input and output
    // grids are the same one, and the snippet in fact rejects an evaluate whose Color and
    // Output rects differ in size. The guide buffers, which may be smaller, are described to
    // the snippet through their own DLSSNR.DepthSubrect*/DLSSNR.MVecSubrect* quadruples (each
    // sized from its own image) and through DLSSNR.MVecScaleX/Y below.
    uint32_t inputSize[2] = { colorExtent.width, colorExtent.height };
    uint32_t outputSize[2] = { colorExtent.width, colorExtent.height };

    if (!m_neuralRenderingContext->initialize(ctx, inputSize, outputSize)) {
      return;
    }

    // NV-DXVK start: DLSS-NR
    // Remix has no paper white of its own, but it does have the scalar that maps scene linear
    // radiance onto display referred values: the tonemapper's exposure. Reuse it so the proxy's
    // soft clip knee follows the scene instead of sitting at a fixed radiance.
    //
    // Note: at this point in the frame the exposure texture holds the PREVIOUS frame's value,
    // because DxvkAutoExposure::dispatch runs from RtxContext::dispatchToneMapping ten lines
    // later. It is heavily temporally smoothed so the lag is not visible, and both codec
    // dispatches read the same texel in the same frame so they can never disagree.
    //
    // Note: createResources() is idempotent and is only called on the DLSS/DLSS-RR/XeSS
    // branches of the upscaler switch, so it has to be called here too --- with NIS, TAA-U or
    // the plain copy nothing would have created the texture before this pass runs.
    DxvkAutoExposure& autoExposure = m_device->getCommon()->metaAutoExposure();
    autoExposure.createResources(ctx);

    const Resources::Resource& exposureTexture = autoExposure.getExposureTexture();

    if (!exposureTexture.isValid()) {
      ONCE(Logger::warn("NVIDIA DLSS-NR skipped: the auto exposure texture the HDR colour codec reads could not be created."));
      return;
    }

    const bool autoExposureEnabled = trackAutoExposure() && autoExposure.enabled();
    const float proxyScale = calcProxyScale();

    // Stage 1 of 3: build the display referred proxy the network actually gets to see.
    dispatchProxyEncode(ctx, rtOutput, exposureTexture.view, autoExposureEnabled, proxyScale, colorExtent);
    // NV-DXVK end

    {
      // Note: when Ray Reconstruction is running it also writes the surface replacement depth and
      // motion vector pair, which describe the surfaces the resolved image actually shows. Prefer
      // those; otherwise use the general purpose pair that every other upscaler consumes.
      // Note: this must be the same predicate that gates writing them --- cb.enableDLSSRR is
      // DxvkRayReconstruction::useRayReconstruction() (rtx_context.cpp:1077, 1392), which also
      // requires RR to be supported. RtxOptions::isRayReconstructionEnabled() alone would hand
      // the snippet never-written buffers, and trips the AliasedResource ownership assert in
      // REMIX_DEVELOPMENT.
      const bool useRayReconstructionGuides = m_device->getCommon()->metaRayReconstruction().useRayReconstruction();

      // NV-DXVK start: DLSS-NR
      // The snippet is fed the PROXY, not m_finalOutput: it is a display referred image network
      // and m_finalOutput is unbounded linear path traced radiance at this point in the frame.
      const Resources::Resource* colorInput = &m_neuralRenderingProxy;
      // NV-DXVK end
      const Resources::Resource* depthInput = useRayReconstructionGuides
        ? &rtOutput.m_primaryDepthDLSSRR.resource(Resources::AccessType::Read)
        : &rtOutput.m_primaryDepth;
      const Resources::Resource* motionVectorInput = useRayReconstructionGuides
        ? &rtOutput.m_primaryScreenSpaceMotionVectorDLSSRR
        : &rtOutput.m_primaryScreenSpaceMotionVector;
      const Resources::Resource* controlMaskInput = useControlMask()
        ? &rtOutput.m_sharedBiasCurrentColorMask.resource(Resources::AccessType::Read)
        : nullptr;

      // Note: Add texture inputs added here to the pInputs array below to properly access the images.
      std::vector<Rc<DxvkImageView>> pInputs = {
        colorInput->view,
        depthInput->view,
        motionVectorInput->view
      };

      if (controlMaskInput != nullptr) {
        pInputs.push_back(controlMaskInput->view);
      }

      std::vector<Rc<DxvkImageView>> pOutputs = {
        m_neuralRenderingOutput.view
      };

      for (auto input : pInputs) {
        if (input == nullptr) {
          continue;
        }

        barriers.accessImage(
          input->image(),
          input->imageSubresources(),
          input->imageInfo().layout,
          input->imageInfo().stages,
          input->imageInfo().access,
          input->imageInfo().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_READ_BIT);

#ifdef REMIX_DEVELOPMENT
        ctx->cacheResourceAliasingImageView(input);
#endif
      }

      for (auto output : pOutputs) {
        barriers.accessImage(
          output->image(),
          output->imageSubresources(),
          output->imageInfo().layout,
          output->imageInfo().stages,
          output->imageInfo().access,
          output->imageInfo().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT);

#ifdef REMIX_DEVELOPMENT
        ctx->cacheResourceAliasingImageView(output);
#endif
      }

      // Note: this also publishes the encode dispatch's write of the proxy to the snippet. The
      // encode ran through DxvkContext::dispatch, which registered that write into this very
      // barrier set, so the flush below covers both it and the transitions queued above.
      barriers.recordCommands(ctx->getCommandList());

      NGXNeuralRenderingContext::NGXBuffers buffers;
      buffers.pColor = colorInput;
      buffers.pDepth = depthInput;
      buffers.pMotionVectors = motionVectorInput;
      buffers.pOutput = &m_neuralRenderingOutput;
      buffers.pControlMask = controlMaskInput;

      NGXNeuralRenderingContext::NGXSettings settings;
      // NV-DXVK start: DLSS-NR
      settings.resetAccumulation = resetHistory || resetGuideHistory;
      // NV-DXVK end
      // Note: Remix writes post perspective divide NDC depth without inverting it, and
      // DxvkDLSS::mInverseDepth (rtx_dlss.h:124) is never assigned anywhere either.
      settings.depthInverted = false;
      // Note: the motion vectors are absolute pixels on the guide grid with the y axis pointing
      // down, which is the convention DLSS uses (mirrors DxvkRayReconstruction::dispatch,
      // rtx_ray_reconstruction.cpp:223). The snippet works on the colour grid, so convert into
      // it. With any upscaler active the two grids differ and this is the ratio between them
      // --- 2.0 for 4K colour over 1080p guides --- which is the arrangement the snippet's
      // kernel launch block is built for: it receives the colour rect, the mvec rect and this
      // scale as three independent values.
      settings.motionVectorScale[0] = guideExtent.width != 0
        ? static_cast<float>(colorExtent.width) / static_cast<float>(guideExtent.width)
        : 1.0f;
      settings.motionVectorScale[1] = guideExtent.height != 0
        ? static_cast<float>(colorExtent.height) / static_cast<float>(guideExtent.height)
        : 1.0f;
      settings.intensity = intensity();
      settings.localToneStrength = localToneStrength();
      settings.localStructureStrength = localStructureStrength();
      settings.skinStructureStrength = skinStructureStrength();
      settings.style = style();
      settings.useAutoMask = useAutoMask();

      const bool evaluated = m_neuralRenderingContext->evaluateNeuralRendering(ctx, buffers, settings);

      // Note: without this the only positive evidence that the feature ran is the absence of an
      // error, and "running correctly" then looks exactly like "silently doing nothing".
      if (evaluated) {
        ++m_evaluateCount;
        if (m_evaluateCount == 1 || m_evaluateCount == 100) {
          Logger::info(str::format(
            "NVIDIA DLSS-NR evaluated (count=", m_evaluateCount,
            ", colour ", colorExtent.width, "x", colorExtent.height,
            ", guides ", guideExtent.width, "x", guideExtent.height, ")"));
        }
      }

      for (auto output : pOutputs) {
        barriers.accessImage(
          output->image(),
          output->imageSubresources(),
          output->imageInfo().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT,
          output->imageInfo().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_READ_BIT);

        ctx->getCommandList()->trackResource<DxvkAccess::None>(output);
        ctx->getCommandList()->trackResource<DxvkAccess::Write>(output->image());
      }

      // NV-DXVK start: DLSS-NR
      // The decode below cannot establish these dependencies itself. The snippet wrote
      // m_neuralRenderingOutput outside DXVK's tracking, so nothing knows the decode's read of
      // it has to wait; and the barrier set was flushed before the evaluate, which cleared the
      // record of the encode's read of m_finalOutput, so nothing knows the decode's write to it
      // is a write-after-read either. Order both explicitly.
      const Rc<DxvkImageView> finalOutputView = rtOutput.m_finalOutput.view(Resources::AccessType::Read);

      barriers.accessImage(
        finalOutputView->image(),
        finalOutputView->imageSubresources(),
        finalOutputView->imageInfo().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        finalOutputView->imageInfo().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

      barriers.recordCommands(ctx->getCommandList());

      if (!evaluated) {
        // The pass target holds nothing usable --- it is whatever the last successful evaluate
        // left there, or the black it was cleared to on creation. Leave m_finalOutput alone:
        // the resolved frame in it is still the correct image to present.
        return;
      }
    }

    // Stage 3 of 3: carry the network's change back onto the untouched HDR original, in place,
    // so no downstream pass has to know this pass exists. This replaces what used to be a
    // straight RGBA copyImage of the neural target over m_finalOutput --- that copy both
    // discarded the HDR range (the network's answer is display referred) and overwrote the
    // alpha channel Remix's transparency and particle compositing depend on.
    dispatchProxyDecode(ctx, rtOutput, exposureTexture.view, autoExposureEnabled, proxyScale, colorExtent);
    // NV-DXVK end
  }

  // NV-DXVK start: DLSS-NR
  float DxvkNeuralRendering::calcProxyScale() {
    // Note: with exposure tracking off this is exactly the RenoDX transform, proxy = colour /
    // paperWhiteScale. With it on, the static half is only the user brightness bias --- the
    // scene adaptive half lives in the auto exposure texture and is applied in the shader.
    //
    // Note: rtx.tonemap.exposureBias is deliberately not folded in. It is private to
    // DxvkToneMapping, defaults to 0.0, and is a global-tonemapper-specific trim; the proxy
    // only needs diffuse white to land near 1.0, not to match the tone curve exactly.
    const float staticExposure = trackAutoExposure()
      ? exp2f(RtxOptions::calcUserEVBias())
      : 1.0f;

    return staticExposure / std::max(paperWhiteScale(), 0.01f);
  }

  void DxvkNeuralRendering::dispatchProxyEncode(
      Rc<RtxContext> ctx,
      const Resources::RaytracingOutput& rtOutput,
      const Rc<DxvkImageView>& exposureView,
      bool autoExposureEnabled,
      float proxyScale,
      const VkExtent3D& colorExtent) {
    ScopedGpuProfileZone(ctx, "DLSS-NR Proxy Encode");

    // Note: without this the push constants are written into the D3D9 bank and the shader reads
    // whatever happens to be there. This pass issued no compute of its own before the codec, so
    // it never needed it.
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    NeuralRenderingEncodeArgs pushArgs = {};
    pushArgs.imageSize = uvec2 { colorExtent.width, colorExtent.height };
    pushArgs.proxyScale = proxyScale;
    pushArgs.enableAutoExposure = autoExposureEnabled ? 1u : 0u;

    const VkExtent3D workgroups = util::computeBlockCount(colorExtent, VkExtent3D { 16, 16, 1 });

    ctx->bindResourceView(NEURAL_RENDERING_ENCODE_COLOR_INPUT, rtOutput.m_finalOutput.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(NEURAL_RENDERING_ENCODE_EXPOSURE_INPUT, exposureView, nullptr);
    ctx->bindResourceView(NEURAL_RENDERING_ENCODE_PROXY_OUTPUT, m_neuralRenderingProxy.view, nullptr);

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, NeuralRenderingEncodeShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void DxvkNeuralRendering::dispatchProxyDecode(
      Rc<RtxContext> ctx,
      const Resources::RaytracingOutput& rtOutput,
      const Rc<DxvkImageView>& exposureView,
      bool autoExposureEnabled,
      float proxyScale,
      const VkExtent3D& colorExtent) {
    ScopedGpuProfileZone(ctx, "DLSS-NR Proxy Decode");

    // Note: the snippet bound its own pipeline on this command buffer, but every bind below
    // marks the compute pipeline, its state and its descriptors dirty (DxvkContext::bindShader
    // sets CpDirtyPipeline/CpDirtyPipelineState/CpDirtyResources), so the dispatch re-issues
    // all of them rather than trusting DXVK's cached view of the command buffer.
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    NeuralRenderingDecodeArgs pushArgs = {};
    pushArgs.imageSize = uvec2 { colorExtent.width, colorExtent.height };
    pushArgs.proxyScale = proxyScale;
    pushArgs.enableAutoExposure = autoExposureEnabled ? 1u : 0u;
    pushArgs.transferStrength = std::clamp(transferStrength(), 0.0f, 1.0f);
    pushArgs.colorStrength = std::clamp(colorStrength(), 0.0f, 1.0f);

    const VkExtent3D workgroups = util::computeBlockCount(colorExtent, VkExtent3D { 16, 16, 1 });

    ctx->bindResourceView(NEURAL_RENDERING_DECODE_PROXY_INPUT, m_neuralRenderingProxy.view, nullptr);
    ctx->bindResourceView(NEURAL_RENDERING_DECODE_NEURAL_INPUT, m_neuralRenderingOutput.view, nullptr);
    ctx->bindResourceView(NEURAL_RENDERING_DECODE_EXPOSURE_INPUT, exposureView, nullptr);
    ctx->bindResourceView(NEURAL_RENDERING_DECODE_COLOR_INPUT_OUTPUT, rtOutput.m_finalOutput.view(Resources::AccessType::ReadWrite), nullptr);

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, NeuralRenderingDecodeShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }
  // NV-DXVK end

  void DxvkNeuralRendering::showImguiSettings() {
    constexpr ImGuiSliderFlags sliderFlags = ImGuiSliderFlags_AlwaysClamp;

    ImGui::BeginDisabled(!supportsNeuralRendering());
    RemixGui::Checkbox("Enable Neural Rendering", &enableObject());
    ImGui::EndDisabled();

    if (!supportsNeuralRendering()) {
      ImGui::Indent();
      ImGui::TextWrapped(str::format("Unavailable: ", getNotSupportedReason()).c_str());
      ImGui::Unindent();
      return;
    }

    ImGui::BeginDisabled(!enable());
    ImGui::Indent();

    // Note: 1.0 is the value the snippet substitutes when the host supplies nothing. The scale
    // these sit on is not documented anywhere and was not recovered from the DLL, so the slider
    // ranges below are a usable authoring range, not a calibrated one.
    RemixGui::DragFloat("Intensity", &intensityObject(), 0.01f, 0.0f, 4.0f, "%.3f", sliderFlags);
    RemixGui::DragFloat("Local Tone Strength", &localToneStrengthObject(), 0.01f, 0.0f, 4.0f, "%.3f", sliderFlags);

    RemixGui::Checkbox("Use Auto Mask", &useAutoMaskObject());
    RemixGui::Checkbox("Use Control Mask", &useControlMaskObject());

    // Both structure strengths are gated by the snippet's auto mask, and binding an explicit
    // control mask turns that auto mask off, so grey them out when they cannot do anything.
    const bool structureStrengthActive = useAutoMask() && !useControlMask();

    ImGui::BeginDisabled(!structureStrengthActive);
    RemixGui::DragFloat("Local Structure Strength", &localStructureStrengthObject(), 0.01f, 0.0f, 4.0f, "%.3f", sliderFlags);
    // Note: any negative value is the snippet's "inherit local structure strength" sentinel, so
    // the range has to reach below zero. 0.0 flattens skin structure, it is not a neutral value.
    RemixGui::DragFloat("Skin Structure Strength", &skinStructureStrengthObject(), 0.01f, -1.0f, 4.0f, "%.3f", sliderFlags);
    ImGui::EndDisabled();

    RemixGui::DragInt("Style", &styleObject(), 0.1f, 0, 15, "%d", sliderFlags);

    // NV-DXVK start: DLSS-NR
    if (RemixGui::CollapsingHeader("HDR Color Codec", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Indent();

      ImGui::TextWrapped("This pass runs before tone mapping, so the colour it is handed is unbounded linear "
                         "path traced radiance while DLSS-NR expects a display referred image. The codec builds "
                         "a soft clipped, sRGB encoded proxy for the network and carries its answer back onto "
                         "the untouched HDR original.");

      // A fixed scale is the RenoDX arrangement; tracking exposure is the Remix native one and
      // is what keeps the proxy's soft clip knee in the right place as the scene brightness
      // changes.
      RemixGui::Checkbox("Track Auto Exposure", &trackAutoExposureObject());
      RemixGui::DragFloat("Paper White Scale", &paperWhiteScaleObject(), 0.01f, 0.01f, 64.0f, "%.3f", sliderFlags);

      // Note: 0.0 here is an exact bypass --- the frame comes out bit for bit unchanged --- which
      // makes this the first thing to reach for when deciding whether an artifact is ours.
      RemixGui::DragFloat("Transfer Strength", &transferStrengthObject(), 0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);
      // Note: 0.0 keeps the original's chromaticity exactly and transfers only the network's
      // luminance change, which is the escape hatch for a colour cast. In pixels dark enough
      // that the original's hue is only path tracer noise the decode fades back to the
      // network's own colour, so turning this down cannot manufacture a coloured speckle in
      // shadows.
      RemixGui::DragFloat("Color Strength", &colorStrengthObject(), 0.01f, 0.0f, 1.0f, "%.2f", sliderFlags);

      ImGui::Unindent();
    }

    // Escape hatch only. Guides at render resolution with colour at output resolution is a
    // supported configuration, and turning this on makes the pass do nothing whenever an
    // upscaler is active.
    RemixGui::Checkbox("Require Matching Guide Resolution", &requireMatchingGuideResolutionObject());
    // NV-DXVK end

    ImGui::Unindent();
    ImGui::EndDisabled();
  }
} // namespace dxvk
