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

#include "rtx/utility/shader_types.h"

// NV-DXVK start: DLSS-NR
// Binding indices for the DLSS-NR HDR colour codec.
//
// The codec exists because DLSS-NR is a display referred image network and this pass runs at
// rtx_context.cpp:748, before tone mapping, where m_finalOutput still holds unbounded linear
// path traced radiance. Handing that to the network is out of distribution and is what the
// "lighting is broken" / "everything's blue and red" reports on an RTX 4090 were. The encode
// pass builds a soft clipped, sRGB encoded proxy for the network, and the decode pass carries
// the network's answer back onto the untouched HDR original.
//
// Modelled on the RenoDX DLSS5 add-on, which is the only known working DLSS-NR deployment.

#define NEURAL_RENDERING_ENCODE_COLOR_INPUT          0
#define NEURAL_RENDERING_ENCODE_EXPOSURE_INPUT       1
#define NEURAL_RENDERING_ENCODE_PROXY_OUTPUT        10

#define NEURAL_RENDERING_DECODE_PROXY_INPUT          0
#define NEURAL_RENDERING_DECODE_NEURAL_INPUT         1
#define NEURAL_RENDERING_DECODE_EXPOSURE_INPUT       2
#define NEURAL_RENDERING_DECODE_COLOR_INPUT_OUTPUT  10

// Push constants

struct NeuralRenderingEncodeArgs {
  // Colour grid the proxy is built on --- always the output (target) extent, because the
  // snippet requires the Color and Output rects to have identical dimensions.
  uvec2 imageSize;
  // Static part of the scene linear -> display referred scale. The auto exposure texture
  // supplies the rest when enableAutoExposure is set; both passes must end up with the same
  // number, so they compute it the same way from the same inputs.
  float proxyScale;
  uint enableAutoExposure;
};

struct NeuralRenderingDecodeArgs {
  uvec2 imageSize;
  float proxyScale;
  uint enableAutoExposure;
  // Global lerp back toward the untouched original. 0 is an exact bypass of the whole pass.
  float transferStrength;
  // 0 keeps the original's chromaticity exactly and transfers only the network's luminance
  // change; 1 takes the network's colour as well.
  float colorStrength;
  uint pad0;
  uint pad1;
};
// NV-DXVK end
