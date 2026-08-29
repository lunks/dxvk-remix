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
#include "rtx/pass/neural_rendering/neural_rendering.h"
#include "rtx/pass/neural_rendering/neural_rendering_codec.slangh"

// NV-DXVK start: DLSS-NR
// Builds the display referred proxy image DLSS-NR actually gets to see.
//
//   proxy = SrgbEncode(SoftClip(max(finalOutput, 0) * exposureScale))
//
// m_finalOutput is R16G16B16A16_SFLOAT and its alpha is meaningful to later passes
// (transparency / particle compositing), so it must survive this pass untouched. It does,
// because nothing here writes m_finalOutput at all and the decode restores the original alpha
// from it verbatim --- see the decode pass.

layout(binding = NEURAL_RENDERING_ENCODE_COLOR_INPUT)
RWTexture2D<float4> InColorBuffer;

layout(binding = NEURAL_RENDERING_ENCODE_EXPOSURE_INPUT)
RWTexture1D<float> InExposure;

layout(binding = NEURAL_RENDERING_ENCODE_PROXY_OUTPUT)
RWTexture2D<float4> OutProxy;

layout(push_constant)
ConstantBuffer<NeuralRenderingEncodeArgs> cb;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= cb.imageSize))
  {
    return;
  }

  const float4 source = InColorBuffer[threadId];

  // Note: the path tracer can leave NaN/Inf in a pixel. Those must not reach the network, and
  // they must not reach the decode's subtraction either, so they are flushed to black here.
  // The decode leaves the corresponding original pixel completely alone.
  const bool sourceIsFinite = !any(isnan(source.rgb)) && !any(isinf(source.rgb));
  const float3 sceneLinear = sourceIsFinite ? max(source.rgb, 0.0f) : float3(0.0f, 0.0f, 0.0f);

  const float scale = neuralRenderingProxyScale(InExposure, cb.enableAutoExposure != 0, cb.proxyScale);

  const float3 displayReferred = sceneLinear * scale;
  const float3 proxy = neuralRenderingSrgbEncode(neuralRenderingSoftClip(displayReferred));

  // Note: the proxy's own alpha is never read back --- the decode takes the alpha it writes
  // from the untouched m_finalOutput --- and DLSS-NR is an RGB network, so a constant is both
  // sufficient and safer here than forwarding m_finalOutput's alpha, which is unbounded and
  // could carry a NaN into the snippet's input texture.
  OutProxy[threadId] = float4(proxy, 1.0f);
}
// NV-DXVK end
