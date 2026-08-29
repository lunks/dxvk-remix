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
#include <vector>

#include "rtx_neural_rendering.h"

#include "rtx.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_imgui.h"
#include "rtx_ngx_neural_rendering.h"
#include "rtx_ray_reconstruction.h"

#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"

namespace dxvk {

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
  }

  void DxvkNeuralRendering::releaseTargetResource() {
    m_neuralRenderingOutput.reset();
  }

  void DxvkNeuralRendering::release() {
    m_neuralRenderingContext = nullptr;
    m_contextCreationFailed = false;
    m_loggedGuideResolutionMismatch = false;
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

    if (!isActive() || !m_neuralRenderingOutput.isValid()) {
      return;
    }

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

    if (requireMatchingGuideResolution() &&
        (colorExtent.width != guideExtent.width || colorExtent.height != guideExtent.height)) {
      if (!m_loggedGuideResolutionMismatch) {
        Logger::warn(str::format("NVIDIA DLSS-NR skipped: render resolution ", guideExtent.width, "x", guideExtent.height,
                                 " does not match output resolution ", colorExtent.width, "x", colorExtent.height,
                                 ", so the depth and motion vector guides are on a different grid than the colour. "
                                 "Set rtx.neuralRendering.requireMatchingGuideResolution to False to run it anyway."));
        m_loggedGuideResolutionMismatch = true;
      }

      return;
    }

    // DLSSNR.InputWidth/InputHeight describe the resource bound as DLSSNR.Color, and that is
    // the already resolved image at output resolution --- this pass does not upscale, so the
    // input and output grids are the same one. The guide buffers, which may be smaller, are
    // described to the snippet through their own DLSSNR.DepthSubrect*/DLSSNR.MVecSubrect*
    // quadruples (each sized from its own image) and through DLSSNR.MVecScaleX/Y below.
    uint32_t inputSize[2] = { colorExtent.width, colorExtent.height };
    uint32_t outputSize[2] = { colorExtent.width, colorExtent.height };

    if (!m_neuralRenderingContext->initialize(ctx, inputSize, outputSize)) {
      return;
    }

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

      const Resources::Resource* colorInput = &rtOutput.m_finalOutput.resource(Resources::AccessType::Read);
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

      barriers.recordCommands(ctx->getCommandList());

      NGXNeuralRenderingContext::NGXBuffers buffers;
      buffers.pColor = colorInput;
      buffers.pDepth = depthInput;
      buffers.pMotionVectors = motionVectorInput;
      buffers.pOutput = &m_neuralRenderingOutput;
      buffers.pControlMask = controlMaskInput;

      NGXNeuralRenderingContext::NGXSettings settings;
      settings.resetAccumulation = resetHistory;
      // Note: Remix writes post perspective divide NDC depth without inverting it, and
      // DxvkDLSS::mInverseDepth (rtx_dlss.h:124) is never assigned anywhere either.
      settings.depthInverted = false;
      // Note: the motion vectors are absolute pixels on the guide grid with the y axis pointing
      // down, which is the convention DLSS uses (mirrors DxvkRayReconstruction::dispatch,
      // rtx_ray_reconstruction.cpp:223). The snippet is told the colour grid, so convert into
      // it. In the default configuration the two grids are identical and this is exactly 1.0;
      // the scaled case only arises with requireMatchingGuideResolution off, which is itself
      // unproven territory.
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

      for (auto output : pOutputs) {
        barriers.accessImage(
          output->image(),
          output->imageSubresources(),
          output->imageInfo().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT,
          output->imageInfo().layout,
          output->imageInfo().stages,
          output->imageInfo().access);

        ctx->getCommandList()->trackResource<DxvkAccess::None>(output);
        ctx->getCommandList()->trackResource<DxvkAccess::Write>(output->image());
      }

      // The copy below cannot establish these dependencies itself: DxvkContext::copyImageHw()
      // only flushes when the barrier set still reports the images as dirty, and
      // recordCommands() clears that tracking on every flush --- so by the time the copy runs
      // the set is empty and it inserts nothing but its own TRANSFER-to-TRANSFER layout
      // transition. Order it here instead. Both images need it: the transfer reads the target
      // the snippet just wrote, and it overwrites m_finalOutput, which the snippet just read as
      // DLSSNR.Color (and copyImageHw transitions a full-subresource destination out of
      // VK_IMAGE_LAYOUT_UNDEFINED, which is free to discard what is still being read).
      barriers.accessImage(
        m_neuralRenderingOutput.view->image(),
        m_neuralRenderingOutput.view->imageSubresources(),
        m_neuralRenderingOutput.view->imageInfo().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        m_neuralRenderingOutput.view->imageInfo().layout,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_READ_BIT);

      barriers.accessImage(
        colorInput->view->image(),
        colorInput->view->imageSubresources(),
        colorInput->view->imageInfo().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        colorInput->view->imageInfo().layout,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT);

      barriers.recordCommands(ctx->getCommandList());

      if (!evaluated) {
        // The pass target holds nothing usable --- it is whatever the last successful evaluate
        // left there, or the black it was cleared to on creation. Leave m_finalOutput alone:
        // the resolved frame in it is still the correct image to present.
        return;
      }
    }

    // Fold the result back into the final output so no downstream pass has to know this pass
    // exists. Note this takes write ownership of the aliased final output, which is legal here
    // because the upscaler already wrote it earlier in the frame.
    ctx->copyImage(
      rtOutput.m_finalOutput.image(Resources::AccessType::Write),
      { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      { 0, 0, 0 },
      m_neuralRenderingOutput.image,
      { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      { 0, 0, 0 },
      colorExtent);
  }

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

    RemixGui::Checkbox("Require Matching Guide Resolution", &requireMatchingGuideResolutionObject());

    ImGui::Unindent();
    ImGui::EndDisabled();
  }
} // namespace dxvk
