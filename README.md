# DLSS-NR for Remix Plus

This build adds **DLSS Neural Rendering** (NVIDIA NGX feature 18, the `nvngx_dlssnr.dll`
snippet) to the Remix Plus runtime as a post-pass that runs after upscaling and sharpening
and before every post-process, including tone mapping.

It is the NVIDIA `dxvk-remix` DLSS-NR integration ported onto Remix Plus so it can be used
by the [Dolphin](https://github.com/Kim2091/dolphin) GameCube/Wii emulator's Remix video
backend, which requires the Remix Plus ABI line (`REMIXAPI_VERSION_MINOR = 1000`) and
rejects a stock NVIDIA runtime with `INCOMPATIBLE_VERSION`. The pass itself is unchanged:
it is exclusively `NVSDK_NGX_VULKAN_*` and sits downstream of the point where the Remix API
path and the D3D9 path converge on the same `RtxContext`, so it behaves identically whether
geometry arrives from D3D9 interception or from `remixapi_*`.

The pass was verified on an **RTX 4090** at driver **610.43.02**, in Portal with RTX on
Linux via Proton — hardware and a driver that NVIDIA's own gates exclude.

## Install

You need three files, and they all go **next to the runtime DLL** — not next to the
executable, if you have pointed the runtime somewhere else. The snippet is located relative
to the calling module, so it must follow `d3d9.dll` wherever that lives.

1. Take `d3d9.dll` and `remix_nvngx.dll` from a `rtx-remix-dlssnr-x64-*` artifact of the
   [Build DLSS-NR workflow](../../actions/workflows/build-dlssnr.yml), or build them
   yourself (see below).
2. Supply your own `nvngx_dlssnr.dll`. **This is not distributed here** — see below.

### For the Dolphin Remix backend

The Dolphin backend loads the runtime by an explicit non-`d3d9` filename, so **rename
`d3d9.dll` to `d3d9-remix.dll`** and place it beside `Dolphin.exe`:

```
<Dolphin.exe dir>\
    Dolphin.exe
    d3d9-remix.dll        <- the Remix Plus runtime, RENAMED from the build's d3d9.dll
    remix_nvngx.dll       <- the DLSS-NR call trampoline, keep this exact name
    nvngx_dlssnr.dll      <- the snippet, user supplied
    <the rest of _output\*.dll and _output\usd\ from the build>
```

The rename is not cosmetic. Qt's `platforms\qwindows.dll` carries a real import on
`d3d9.dll`, so a runtime with that name is dragged in and fully self-initialised while Qt
is still starting, about 1.8 seconds before the video backend runs — which breaks per-game
config files and loads the path tracer into every Dolphin process. It also does not
interfere with the trampoline: the caller check looks for the substring `nvngx.dll` in the
calling module's path, and `d3d9-remix.dll` does not contain it, so the trampoline path is
taken normally.

Use the **x64** artifact. Do not use a `.trex/` layout or `bridge/_output/d3d9.dll` — that
is the 32-bit bridge client and Dolphin is x64.

### Enabling it

Either in the game's `rtx.conf` (for Dolphin, that is `Remix\<GameID>\rtx.conf`):

```ini
rtx.neuralRendering.enable = True
```

or by exporting `RTX_NEURAL_RENDERING_ENABLE=1` before launching, which beats `rtx.conf`
and needs no file plumbing — the easiest first smoke test. It can also be toggled in the
Remix developer menu (`Alt+X`) under **Neural Rendering**, but note that this option is a
`UserSetting`, so the dev menu persists it to `user.conf`; this fork does not yet honour
`DXVK_USER_CONFIG_FILE`, so under Dolphin that file is shared across all games and wins
over any per-game `rtx.conf`. Prefer the per-game `rtx.conf` or the environment variable.

`remix_nvngx.dll` is required, not optional, and must keep that exact filename — every
gated snippet export resolves its caller's module from the return address and requires the
substring `nvngx.dll` in that path, and this module is what satisfies it. Without it,
initialisation fails with `0xbad00002` and the feature reports itself unsupported.

### The snippet

`nvngx_dlssnr.dll` is NVIDIA's proprietary DLL and is deliberately **not** included here.
You must supply your own. It needs to contain **sm_89** cubins: the stock snippet ships
`sm_120` (Blackwell) only, so on Ada it will load, pass the checks, and then fail at kernel
load. The network's kernels use FP8 (`e4m3`) tensor ops, so **Ada or newer is a hard
floor** — Ampere and earlier cannot run it at all, regardless of patching. See
`tools/patch_dlssnr.py`.

## Check that it is actually running

**The frame looks identical whether or not the pass ran**, so the absence of errors proves
nothing. The runtime log is the only reliable confirmation. Under Dolphin it is at
`Remix\<GameID>\rtx-remix\logs\remix-dxvk.log` — *not* `dolphin.log`. Look for:

```
NVIDIA DLSS-NR snippet loaded from ...\nvngx_dlssnr.dll (via remix_nvngx.dll)
NVIDIA DLSS-NR evaluated (count=1, colour 3840x2160, guides 1920x1080)
NVIDIA DLSS-NR evaluated (count=100, colour 3840x2160, guides 1920x1080)
```

`colour` is the output grid and `guides` the render grid; they differ when an upscaler is
active, which is normal and supported. `(direct)` in place of `(via remix_nvngx.dll)` means
the trampoline was not found next to the runtime. If you instead see a line beginning
`NVIDIA DLSS-NR skipped:` or `NVIDIA DLSS-NR inactive:`, it states the reason; each distinct
reason is reported once.

## Settings

All live under `rtx.neuralRendering.` and appear in the developer menu.

| Option | Default | Notes |
|---|---|---|
| `enable` | `False` | Also settable via `RTX_NEURAL_RENDERING_ENABLE` |
| `intensity` | `1.0` | |
| `localToneStrength` | `1.0` | |
| `localStructureStrength` | `1.0` | Inert unless `useAutoMask` is on and `useControlMask` is off |
| `skinStructureStrength` | `-1.0` | Negative means "inherit `localStructureStrength`". `0.0` is **not** neutral — it flattens skin structure |
| `style` | `0` | |
| `useAutoMask` | `True` | Gates **both** structure strengths; forced off when `useControlMask` is on |
| `useControlMask` | `False` | Binds an explicit control mask, which disables the auto mask |
| `requireMatchingGuideResolution` | `False` | Escape hatch: set `True` to skip the pass when colour and guides differ |
| `paperWhiteScale` | `1.0` | HDR codec; raise if the proxy looks blown out, lower if it looks black |
| `trackAutoExposure` | `True` | Folds auto exposure into the proxy scale |
| `transferStrength` | `1.0` | `0.0` is an exact bypass, returned bit for bit |
| `colorStrength` | `1.0` | Lower if the image picks up a colour cast |

`1.0` is the value the snippet falls back to when the host supplies nothing. It is **not** a
calibrated neutral midpoint, and these knobs have not been characterised — do not assume
`2.0` is "double".

There is deliberately no preset selector: the snippet ships exactly one network
(`CC_SILVER_AARDWOLD`, preset 1) and every other preset value falls back to it.

## Building

`.github/workflows/build-dlssnr.yml` builds this on the same Windows runner and toolchain
as Remix Plus's own `build.yml`, and uploads both `d3d9.dll` and `remix_nvngx.dll`. Locally:

```powershell
. .\build_common.ps1
PerformBuild -BuildFlavour release -BuildSubDir _Comp64Release -Backend ninja -EnableTracy false
```

Both DLLs land in `_output\`. Remix Plus build instructions are below.

---

# Remix Plus

[![Build Status](https://github.com/RemixProjGroup/dxvk-remix/actions/workflows/build.yml/badge.svg)](https://github.com/RemixProjGroup/dxvk-remix/actions/workflows/build.yml)

**Remix Plus** is a community-maintained fork of NVIDIA's
[`dxvk-remix`](https://github.com/NVIDIAGameWorks/dxvk-remix) — created
and led by [Kim2091](https://github.com/Kim2091) — that extends the
Remix SDK API for modern-game plugin integrations. It
brings the SDK extensions developed in the gmod-rtx community fork —
batched mesh and light creation, plugin-injected game state, UI state
plumbing, VRAM control, additional tonemap operators, the Numos
sky system, and more — onto a clean, NVIDIA-rebase-friendly
base, so plugin authors and game integrations can build on a
maintained codebase that's API-compatible with the broader Remix
ecosystem.

Like upstream `dxvk-remix`, Remix Plus is a fork of
[DXVK](https://github.com/doitsujin/dxvk) that overhauls the D3D9
fixed-function pipeline for path-traced remastering. The `bridge`
subfolder enables 32-bit games to communicate with the 64-bit
runtime.

> Bugs encountered with Remix Plus belong in this repo's issue
> tracker, not in upstream DXVK or NVIDIA `dxvk-remix`.

## What's new vs upstream `dxvk-remix`

### Remix SDK API extensions

- **Batched mesh creation** — `CreateMeshBatched` for high-throughput
  geometry submission paths.
- **Batched light creation + deferred updates** — `CreateLightBatched`,
  `UpdateLightDefinition` for per-frame light churn.
- **UI state query/set** — `GetUIState` / `SetUIState` so plugins can
  observe and drive Remix's developer UI from outside the runtime.
- **Texture-hash category mutation** — `AddTextureHash`,
  `RemoveTextureHash`, `dxvk_GetTextureHash` for plugin-driven texture
  classification at runtime.
- **D3D11 shared-texture handles** — `dxvk_GetSharedD3D11TextureHandle`
  for interop with D3D11-side rendering paths.
- **VRAM control** — `RequestTextureVramFree`, `RequestVramCompaction`,
  `GetVramStats` give plugins a driver-view handle on memory pressure.
- **Plugin-injected game state** — `SetGameValue` writes named values
  into a fork-owned store; `GameValueReadBool` and `GameValueReadNumber`
  graph (Sense) components read them back inside replacement logic.
- **`externalMesh` field** on `RasterGeometry` for capture/replacement
  parity when geometry comes in via the Remix API path.
- **`InstanceCategoryBit` ABI** synced to the gmod/plugin layout so
  category bits round-trip correctly across the API boundary.

### Tonemapping & auto-exposure

- **Eight tonemap operators** in the UI dropdown: Hill ACES, Narkowicz
  ACES, Hable Filmic, AgX Minimal, Lottes 2016, PsychoV17_Beta, Gran
  Turismo 7 (SDR), and Neutwo. Each operator has its own parameter
  panel — controls are visible at a glance instead of buried.
- **AgX Minimal** (Benjamin Wrensch / MIT) replaces the older
  multi-knob AgX surface. Look presets: None / Golden / Punchy.
- **PsychoV17_Beta** — Slang port of renodx Psycho Test 17 (Carlos
  Lopez Jr. / MIT). Stockman-Sharpe LMS + Naka-Rushton cone response
  + gamut compression.
- **Gran Turismo 7 reference** — Slang port of Polyphony Digital's
  SIGGRAPH 2025 GT7 tone-mapping reference (MIT). SDR mode, ICtCp UCS.
- **Hable presets** (Hejl, Uncharted 2) with the original parameters.
- **Perceptual auto-exposure** — Stockman-Sharpe Yf histogram +
  geometric-mean adaptation + first-site cone-contrast law. Asymmetric
  in log-exposure space: cone-bleach is fast (~0.10–0.20 s),
  rod-recovery is slow (~0.50–1.50 s). Two tau sliders replace the old
  Adaptation Speed / EV-Min / EV-Max / Average Mode controls.

The legacy Tonemapping Mode (Global / Local / Direct) combo, the local
tonemapper, the dynamic tone curve / Tuning Mode sliders, the User
Brightness slider, and the exposure-compensation curve are removed —
the apply pass always runs in operator-only mode.

### Numos sky system

- **Numos atmosphere (Hillaire scattering)** — physically-based atmospheric scattering
  ported from the gmod-rtx community fork. Daylight, sunset, and
  twilight all behave correctly without manual fog tuning.
- **Volumetric clouds** — procedural FBM cloud layer with
  weather-driven coverage and Nubis-style spatial variation. Anvil,
  shear, and vertical-profile shaping are artist-tunable. Renders
  through a sky-dome curvature with sample-seam jitter to hide
  stepping artifacts. Sun and any number of moons cast shadows
  through the volume; twilight and night cloud lighting are
  physically correct rather than tuned-by-eye. Shadow-tap cost is
  heavily reduced via multi-octave density approximation,
  cadence-decoupled shadow caching, combined-moon marching, and
  density-gated skipping.
- **Night sky** — stars, milky way, shooting stars, and airglow,
  with sidereal rotation so the celestial sphere actually moves.
  Multi-moon support: independent elevation / rotation / phase per
  moon, unified moon-disk eval with surface-style presets (Rocky,
  Volcanic), soft radial glow/halo, and physically-scaled lunar
  illumination on the cloud volume.

### Hardware skinning

- **HW skinning** with capture and replacement parity, so skinned
  meshes injected via the Remix API path participate in capture and
  asset replacement the same as fixed-pipeline geometry.

### Capture and overlay quality-of-life

- **Overwrite-existing-capture** checkbox in the capture dialog.
- **Null-image / null-map / dimension guards** on capture export
  paths — eliminates a class of crashes when capturing edge-case
  resources.
- **Keyboard and mouse events** forwarded to ImGui on the legacy
  `WndProc` fallback path, so plugin-API-driven overlays receive
  input even when a game menu captures raw input.
- **Quieter logs** — spammy swapchain-recreate throws and repeated
  mesh-registration warnings silenced.

### Engineering

- **Fork-touchpoint pattern** — fork logic is extracted into
  dedicated `rtx_fork_*.cpp` modules, with one-line dispatches in
  upstream files. Reduces NVIDIA-rebase pain by ~54% (measured) and
  makes the fork's surface area auditable. See
  [`docs/fork-touchpoints.md`](docs/fork-touchpoints.md) for the
  authoritative inventory.
- **PR template fridge-list reminder** keeps the discipline honest.

## Contributing

Contributions are welcome. Whether you write Remix plugins, ship a
game integration, or want to make this fork better — start with the
contribution guide:

**[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)** covers setup,
build, fork-touchpoint discipline, code style, and PR submission.

The short version:

1. Fork [`RemixProjGroup/dxvk-remix`](https://github.com/RemixProjGroup/dxvk-remix).
2. Branch on your fork — any name is fine.
3. Keep PRs small and focused.
4. Build clean (release flavor, exit code 0, zero errors).
5. Open a PR against canonical's `main` branch.
6. Add yourself to `src/dxvk/imgui/dxvk_imgui_about.cpp` under
   "Github Contributors".

If you touch any upstream file, update
[`docs/fork-touchpoints.md`](docs/fork-touchpoints.md) in the same
commit — that's the one rigid rule.

Questions? File an issue or ask on the
[RTX Remix Discord](https://discord.gg/c7J6gUhXMk).

## Quick build

Detailed requirements and walkthrough live in
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md). The compressed
version, assuming you have Visual Studio 2019 (with the v142
toolchain), the Windows SDK, Meson 1.8.2+, the Vulkan SDK
1.4.313.2+, and Python 3.9+:

```powershell
git clone --recursive https://github.com/<your-fork>/dxvk-remix.git
cd dxvk-remix
.\scripts\build.ps1
```

`scripts/build.ps1` is the fork-side runtime build entry point — it
discovers Visual Studio via vswhere, runs `meson setup`/`compile`/
`install`, and verifies artifacts. It defaults to the `release`
flavor; pass `-Flavor debug` or `-Flavor debugoptimized` for the
instrumented flavors, `-Clean` for a fresh build dir, or
`-EnableTracy` for the Tracy profiler.

Output `d3d9.dll` lands in `_Comp64Release/src/d3d9/` and is
installed to `_output/`. Configure game targets via
`gametargets.conf` (copy `gametargets.example.conf`) and the build
will deploy automatically.

To build the 32-bit-to-64-bit bridge separately, use the fork-side
bridge wrapper `.\bridge\scripts\build.ps1` (builds the x64 server
and x86 client + launcher into `bridge/_output/`).

## Remix API

If you're integrating Remix into a game with available source, you
can either use the D3D9 surface directly (Remix's `d3d9.dll`
implements D3D9) or program against the Remix C API to push game
data into the renderer. Start with
[`docs/RemixSDK.md`](docs/RemixSDK.md) for setup and the mental
model, then see [`docs/RemixApi.md`](docs/RemixApi.md) for the full
API reference. The C header is
[`public/include/remix/remix_c.h`](public/include/remix/remix_c.h),
with a type-safe C++ wrapper at
[`public/include/remix/remix.h`](public/include/remix/remix.h).

## Project documentation

- [Anti-Culling System](docs/AntiCullingSystem.md)
- [Cloud System](docs/CloudSystem.md)
- [Contributing Guide](docs/CONTRIBUTING.md)
- [Contributing Style Guide](docs/CONTRIBUTING-style-guide.md)
- [Foliage System](docs/FoliageSystem.md)
- [Fork Touchpoints](docs/fork-touchpoints.md)
- [GPU Print](docs/GpuPrint.md)
- [Opacity Micromap](docs/OpacityMicromap.md)
- [Remix API (hub reference)](docs/RemixApi.md)
- [Remix API Changelog](docs/RemixApiChangelog.md)
- [Remix API Surface (auto-generated)](RemixApiSurface.md)
- [Remix Config](docs/RemixConfig.md)
- [Remix Logic](docs/RemixLogic.md)
- [Remix SDK Setup](docs/RemixSDK.md)
- [Remix Sky API](docs/RemixSkyAPI.md)
- [Rtx Options](RtxOptions.md)
- [Terrain System](docs/TerrainSystem.md)
- [Unit Test](docs/UnitTest.md)

## Team

- [Kim2091](https://github.com/Kim2091) — project lead and lead maintainer
- [CR](https://github.com/sambow23) — maintainer
- [TheGreatHMMMM](https://github.com/TheGreatHMMMM) — contributor
- [Gokuwashere](https://github.com/BrunchyChineapple) — contributor

## Credits

Remix Plus stands on the work of:

- [DXVK](https://github.com/doitsujin/dxvk) — D3D9 → Vulkan
  translation layer.
- [NVIDIA `dxvk-remix`](https://github.com/NVIDIAGameWorks/dxvk-remix) —
  path-traced remastering fork of DXVK.
- The **gmod-rtx community fork** — origin of most of the SDK
  extensions Remix Plus carries.

Thanks to all the contributors whose work makes this possible.
