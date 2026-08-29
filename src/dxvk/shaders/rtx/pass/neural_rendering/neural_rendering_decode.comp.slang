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
// Carries DLSS-NR's answer back onto the untouched linear HDR original.
//
// ---------------------------------------------------------------------------------------------
// Derivation of the transfer (this is the correctness critical part of the pass)
// ---------------------------------------------------------------------------------------------
//
// Let s be the scene linear -> display referred scale, and per channel let
//
//   o  = finalOutput                       scene linear radiance, unbounded, >= 0
//   d  = o * s                             display referred original
//   p  = SrgbDecode(Proxy)   == SoftClip(d)  display referred, in [0, 1)
//   n  = SrgbDecode(Neural)                display referred, in [0, 1), what the network returned
//
// The encode threw away exactly (d - p): the part of the original the soft clip could not
// represent. Above the knee that is the entire HDR headroom, and it can be enormous (d = 100
// while p is pinned just under 1.0). Below the knee SoftClip is the identity, so p == d and the
// discarded residual is exactly zero.
//
// The network's answer is a DISPLAY REFERRED image, so the meaningful thing it produced is the
// change it made, (n - p), in display referred units. Put the discarded residual back and the
// whole transfer is one line:
//
//   result = o + (n - p) / s                                                       (*)
//
// Properties, all algebraic:
//
//   1. EXACT IDENTITY. If the network changed nothing, n == p, so (n - p) is exactly +0.0 and
//      (*) evaluates to o + 0.0 == o, bit for bit, for every pixel and every value of s. This
//      is the property that matters most: a no-op network must produce a no-op frame. Note the
//      residual is never materialised as a separate quantity, so there is no
//      "d - p + n, then divide" round trip to lose it in.
//
//   2. HDR SURVIVES. o is never scaled, clipped or reconstructed --- it is added to. A pixel at
//      radiance 100 stays at radiance ~100, so bloom (rtx_context.cpp:753), motion blur (756)
//      and the tone curve (758, which spans 2^-24 .. 2^8 in log2 linear) still see the dynamic
//      range they need. A naive SrgbDecode(neural) / s would have clamped the frame to the
//      soft clip's SDR range and destroyed all of it.
//
//   3. NO SINGULARITIES. (*) has one division, by the push constant scale, which is clamped
//      away from zero in neuralRenderingProxyScale. There is nothing of the form neural/proxy,
//      so a proxy channel sitting at zero (a saturated primary, a black pixel) cannot produce
//      an infinity or an amplified quantisation step. Note this covers (*) only --- the
//      chromaticity valve below carries a second division, by the ORIGINAL's luminance, and the
//      comment where it is written explains what bounds that one.
//
//   4. BELOW THE KNEE IT IS THE NETWORK'S ANSWER VERBATIM. There p == d == o * s, so
//      (*) reduces to o + (n - o*s)/s == n / s: the neural result decoded straight back into
//      scene linear. So in the bulk of a correctly exposed frame the pass really is DLSS-NR,
//      not a diluted version of it.
//
//   5. IN CLIPPED HIGHLIGHTS THE NETWORK'S CHANGE IS ADDITIVE, NOT MULTIPLICATIVE. This is
//      deliberate. o = 100, p = 0.999, and the network answering n = 0.7 gives 99.7, not 70.
//      A multiplicative carry-back would have amplified a display referred change by 100x and
//      blown out the highlight. RenoDX's UpgradeToneMap does the same thing by a different
//      route: its bright branch is new_y = ny + max(0, oy - py), i.e. the clipped headroom
//      re-added additively on top of the neural luminance.
//
// Relationship to RenoDX's UpgradeToneMap. RenoDX works in luminance and then rebuilds the
// colour from the NEURAL pixel (HueOkLab(neural * ratio, neural)); its luminance target reduces
// to the original's luminance when n == p, but its chromaticity does not reduce to the
// original's, so its full chain is only an exact identity when its ColorStrength is 0 or the
// pixel is below the knee. Working per channel on the residual instead makes the identity hold
// unconditionally, and drops the OkLab hue restoration and the AP1 clamp along with it --- both
// of those exist to repair a colour that has been multiplied by a ratio that can reach 100x,
// and (*) never multiplies the colour by anything. The chromaticity valve below does, but only
// at colorStrength < 1, and it is bounded and faded out where a ratio would misbehave rather
// than repaired after the fact.
//
// Two of RenoDX's knobs are worth keeping and are kept:
//
//   colorStrength   0 rebuilds the result from the ORIGINAL's chromaticity carrying only the
//                   network's luminance change, 1 takes the network's colour too. This is the
//                   escape hatch for the "everything's blue/red" class of symptom.
//   transferStrength  a global lerp back toward the untouched original; 0 is an exact bypass.
//
// Both preserve property 1: lerp(x, x, t) == x exactly, and at n == p the luminance ratio is
// exactly 1.0.
// ---------------------------------------------------------------------------------------------

layout(binding = NEURAL_RENDERING_DECODE_PROXY_INPUT)
RWTexture2D<float4> InProxy;

layout(binding = NEURAL_RENDERING_DECODE_NEURAL_INPUT)
RWTexture2D<float4> InNeural;

layout(binding = NEURAL_RENDERING_DECODE_EXPOSURE_INPUT)
RWTexture1D<float> InExposure;

layout(binding = NEURAL_RENDERING_DECODE_COLOR_INPUT_OUTPUT)
RWTexture2D<float4> InOutColorBuffer;

layout(push_constant)
ConstantBuffer<NeuralRenderingDecodeArgs> cb;

// Note: m_finalOutput is R16G16B16A16_SFLOAT. Writing a value the format cannot hold turns the
// pixel into an infinity, which then poisons the bloom downsample and the exposure histogram.
static const float kNeuralRenderingMaxHalf = 65504.0f;

// Below this DISPLAY REFERRED luminance the original's chromaticity stops carrying usable
// information and the chromaticity preserving path is faded out; see the valve in main(). A
// thousandth of diffuse white encodes to an sRGB code value of about 3/255, which is well inside
// what any viewer reads as black. It is expressed display referred and converted into the scene
// linear units `original` is in using the same scale the rest of the codec uses, so it tracks
// the exposure instead of sitting at a fixed radiance.
static const float kNeuralRenderingMinChromaLuminance = 0.001f;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= cb.imageSize))
  {
    return;
  }

  const float4 source = InOutColorBuffer[threadId];

  // A pixel the path tracer already broke is left exactly as it was --- the encode fed the
  // network black for it, so its neural answer means nothing here.
  if (any(isnan(source.rgb)) || any(isinf(source.rgb)))
  {
    return;
  }

  // Note: the max() is what makes the identity argument above hold only for a non-negative
  // original. m_finalOutput is a sum of radiances and is non-negative by construction, so in
  // practice this is the identity; a negative channel would be clamped to zero, which is the
  // right thing to do to it anyway.
  const float3 original = max(source.rgb, 0.0f);

  const float scale = neuralRenderingProxyScale(InExposure, cb.enableAutoExposure != 0, cb.proxyScale);

  const float3 proxy = neuralRenderingSrgbDecode(InProxy[threadId].rgb);
  const float3 neural = neuralRenderingSrgbDecode(InNeural[threadId].rgb);

  // (*) --- the scene linear change the network asked for. Exactly zero when it asked for none.
  const float3 neuralDelta = (neural - proxy) / scale;

  // A non-finite delta can only come from a resource the snippet left in an unexpected state.
  // Never let one reach m_finalOutput: NaN there survives every downstream pass and poisons the
  // exposure histogram and the bloom chain for the rest of the frame.
  if (any(isnan(neuralDelta)) || any(isinf(neuralDelta)))
  {
    return;
  }

  const float3 transferred = max(original + neuralDelta, 0.0f);

  // Chroma safety valve: reproduce the transferred luminance on the ORIGINAL's chromaticity.
  //
  // This is the one place in the pass that divides by something that can go to zero. What comes
  // out of `original * (transferredLuminance / originalLuminance)` always has luminance exactly
  // transferredLuminance, so the RESULT is bounded no matter how small the denominator gets ---
  // but its individual channels are not bounded by that luminance. Reaching a given luminance on
  // a fully saturated blue needs 1/0.0722 = 13.8x that luminance in the blue channel (4.7x for
  // red, 1.4x for green), so a near-black pixel whose hue is path tracer noise comes back as a
  // saturated speckle where the network asked for near neutral dark grey. In a path tracer the
  // near-black pixels are exactly the hue-random ones, so that is a whole region of the frame,
  // not a stray pixel.
  //
  // The previous hard `originalLuminance > 0.0f` branch also made it a discontinuity: a pixel at
  // exactly zero took the network's colour while one a single denormal above took a fully
  // saturated primary. Fade the chromaticity preserving path out instead, over the floor above.
  //
  // Property 1 still holds EXACTLY. At neuralDelta == 0 we have transferred == original and
  // transferredLuminance == originalLuminance, so luminanceRatio is exactly 1.0,
  // original * 1.0 == original, and lerp(original, original, w) == original for every w ---
  // including inside the faded region, where both ends of the lerp are `original`.
  const float originalLuminance = calcBt709Luminance(original);
  const float transferredLuminance = calcBt709Luminance(transferred);

  // Note: scale is clamped to [1e-6, 1e6] by neuralRenderingProxyScale, so chromaFloor lands in
  // [1e-9, 1e3] and can never be zero.
  const float chromaFloor = kNeuralRenderingMinChromaLuminance / scale;
  const float chromaWeight = saturate(originalLuminance / chromaFloor);
  // Note: the guard is still needed even though chromaWeight is zero when it trips --- 0/0 is a
  // NaN and lerp() would propagate it through the zero weight.
  const float luminanceRatio = originalLuminance > 0.0f
    ? transferredLuminance / originalLuminance
    : 1.0f;
  const float3 luminanceOnly = lerp(transferred, original * luminanceRatio, chromaWeight);

  const float3 graded = lerp(luminanceOnly, transferred, cb.colorStrength);
  const float3 result = lerp(original, graded, cb.transferStrength);

  InOutColorBuffer[threadId] = float4(
    clamp(result, 0.0f, kNeuralRenderingMaxHalf),
    // Note: DLSS-NR is an RGB network and whatever it leaves in alpha is not meaningful. The
    // full RGBA copyImage this pass used to end with overwrote Remix's alpha with it, which is
    // what destroyed portal particle transparency. Carry the original alpha through instead.
    source.a);
}
// NV-DXVK end
