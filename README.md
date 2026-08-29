# DLSS-NR for RTX Remix

This fork adds **DLSS Neural Rendering** (NVIDIA NGX feature 18, the `nvngx_dlssnr.dll`
snippet) to the RTX Remix runtime as a post-pass that runs after upscaling and before
tone mapping.

Verified running in Portal with RTX on an **RTX 4090** at driver **610.43.02**, on Linux
via Proton — hardware and a driver that NVIDIA's own gates exclude.

## Install

1. Download `d3d9.dll` and `remix_nvngx.dll` from the
   [latest release](../../releases/latest).
2. Back up your existing `d3d9.dll`, then drop **both** files into the game's Remix
   runtime folder, next to the one you replaced (for Portal with RTX that is
   `PortalRTX/bin/.trex/`).
3. Put a DLSS-NR snippet named `nvngx_dlssnr.dll` in that same folder. **This is not
   distributed here** — see below.
4. Enable it, either in `rtx.conf` / `user.conf`:

   ```ini
   rtx.neuralRendering.enable = True
   ```

   or in the Remix developer menu (`Alt+X`) under **Neural Rendering**.

`remix_nvngx.dll` is required, not optional, and must keep that exact filename — every
gated snippet export checks that its caller's module path contains the substring
`nvngx.dll`, and this module is what satisfies that. Without it, initialisation fails
with `0xbad00002` and the feature reports itself unsupported.

### The snippet

`nvngx_dlssnr.dll` is NVIDIA's proprietary DLL and is deliberately **not** included in
these releases. You must supply your own. It needs to contain **sm_89** cubins: the
stock snippet ships `sm_120` (Blackwell) only, so on Ada it will load, pass the checks,
and then fail at kernel load. The network's kernels use FP8 (`e4m3`) tensor ops, so
**Ada or newer is a hard floor** — Ampere and earlier cannot run it at all, regardless
of patching.

## Check that it is actually running

The runtime log (`rtx-remix/logs/remix-dxvk.log`) is the only reliable confirmation:

```
NVIDIA DLSS-NR snippet loaded from ...
vngx_dlssnr.dll (via remix_nvngx.dll)
NVIDIA DLSS-NR evaluated (count=1, colour 3840x2160, guides 1920x1080)
NVIDIA DLSS-NR evaluated (count=100, colour 3840x2160, guides 1920x1080)
```

`colour` is the output grid and `guides` the render grid; they differ when an upscaler is
active, which is normal and supported. If you instead see a line beginning
`NVIDIA DLSS-NR skipped:` or `NVIDIA DLSS-NR inactive:`, it states the reason.

Note the frame can look completely correct while the pass is doing nothing at all, so
treat the absence of errors as inconclusive and check for an `evaluated` line.

## Settings

All live under `rtx.neuralRendering.` and appear in the developer menu.

| Option | Default | Notes |
|---|---|---|
| `enable` | `False` | |
| `intensity` | `1.0` | |
| `localToneStrength` | `1.0` | |
| `localStructureStrength` | `1.0` | Inert unless `useAutoMask` is on and `useControlMask` is off |
| `skinStructureStrength` | `-1.0` | Negative means "inherit `localStructureStrength`". `0.0` is **not** neutral — it flattens skin structure |
| `style` | `0` | |
| `useAutoMask` | `True` | Gates **both** structure strengths |
| `paperWhiteScale` | `1.0` | HDR codec; raise to brighten what the network sees |
| `requireMatchingGuideResolution` | `False` | Escape hatch: set `True` to skip the pass when colour and guides differ |

`1.0` is the value the snippet falls back to when the host supplies nothing. It is **not**
a calibrated neutral midpoint, and these knobs have not been characterised — do not assume
`2.0` is "double".

There is deliberately no preset selector: the snippet ships exactly one network
(`CC_SILVER_AARDWOLD`, preset 1) and every other preset value falls back to it.

## Building

`.github/workflows/build-dlssnr.yml` builds this on a Windows runner and uploads
`d3d9.dll` and `remix_nvngx.dll`. Upstream build instructions are below.

---

# dxvk-remix

[![Build Status](https://github.com/NVIDIAGameWorks/dxvk-remix/actions/workflows/build.yml/badge.svg)](https://github.com/NVIDIAGameWorks/dxvk-remix/actions/workflows/build.yml)

dxvk-remix is a fork of the [DXVK](https://github.com/doitsujin/dxvk) project, which overhauls the fixed-function graphics pipeline implementation in order to remaster games with path tracing.

Thanks to all the contributors to DXVK for creating this foundational piece of software, on top of which we were able to build the RTX Remix Runtime.

While dxvk-remix is a fork of DXVK, please report bugs encountered with dxvk-remix to this repo rather than to the DXVK project.

dxvk-remix also contains a subproject in the `bridge` folder, which enables 32 bit games to communicate with the 64 bit dxvk-remix runtime.

## Build instructions

### Requirements:
1. Windows 10 or 11
2. [Git](https://git-scm.com/download/win)
3. [Visual Studio ](https://visualstudio.microsoft.com/vs/older-downloads/)
    - VS 2019 is tested
    - VS 2022 may also work, but it is not actively tested
    - Note that our build system will always use the most recent version available on the system
4. [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive/)
    - 10.0.19041.0 is tested
5. [Meson](https://mesonbuild.com/)
    - 1.8.2 has been tested
    - Follow [instructions](https://mesonbuild.com/SimpleStart.html#installing-meson) on how to install and reboot the PC before moving on (Meson will indicate as much)
6. [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows)
    - 1.4.313.2 or newer
    - You may need to uninstall previous SDK if you have an old version
7. [Python](https://www.python.org/downloads/)
    - 3.9 or newer
    - Ensure you are using python installed from the link above and not from the Microsoft Store
    - Python is required by developer build tooling; the packaged RTX Remix Runtime does not link against Python.
8. [DirectX Runtime](https://www.microsoft.com/en-us/download/details.aspx?id=35)
    - Latest version should work.
    - This includes d3d9x*.dll which are required to run the game
    - May already be installed if you have D3D9 games installed

#### Additional notes:
- If dependency paths change (for example, after installing a new Vulkan SDK), reconfigure the affected build from the repository root, such as `meson setup --reconfigure _Comp64Release`.

### Generate and build dxvk-remix Visual Studio project 
1. Clone the repository with all submodules:
	- `git clone --recursive https://github.com/NVIDIAGameWorks/dxvk-remix.git`

	If the clone was made non-recursively and the submodules are missing, clone them separately:
	- `git submodule update --init --recursive`

2. Install all the [requirements](#requirements) before proceeding further

3. Make sure PowerShell scripts are enabled
    - One-time system setup: run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned` in an elevated PowerShell prompt, then close and reopen any existing PowerShell prompts
	
4. To generate and build dxvk-remix project:
    - Right Click on `dxvk-remix\build_dxvk_all_ninja.ps1` and select "Run with Powershell"
    - If that fails or has problems, run the build manually in a way you can read the errors:
        - open a windows file explorer to the `dxvk-remix` folder
        - remove only the generated configuration that failed, such as `_Comp64Debug/`; remove `_vs/` as well only if the generated Visual Studio solution must be recreated
        - type `cmd` in the address bar to open a command line window in that folder.
        - copy and paste `powershell -command "& .\build_dxvk_all_ninja.ps1"` into the command line, then press enter
    - Optional flags:
        - `-SkipApics` — skip downloading game test captures (requires auth token)
    - Examples:
        ```powershell
        .\build_dxvk_all_ninja.ps1
        .\build_dxvk_all_ninja.ps1 -SkipApics
        ```
    - This will build all 3 configurations of dxvk-remix project inside subdirectories of the build tree:
        - **_Comp64Debug** - full debug instrumentation, runtime speed may be slow
        - **_Comp64DebugOptimized** - partial debug instrumentation (i.e. asserts), runtime speed is generally comparable to that of release configuration
        - **_Comp64Release** - fastest runtime
    - This will generate a project in the **_vs** subdirectory
    - This script builds the officially supported x64 targets. ARM64 and ARM64EC configurations are compile-tested in CI but are not part of this local build workflow.

5. Open **_vs/dxvk-remix.sln** in Visual Studio (2019+). 
    - Do not convert the solution on load if prompted when using a newer version of Visual Studio 
    - Once generated, the project can be built via Visual Studio or via powershell scripts
    - A build will copy generated DXVK DLLs to any target project as specified in **gametargets.conf** (see its [setup section](#deploy-built-binaries-to-a-game))

### Deploy built binaries to a game 
1. First time only: copy **gametargets.example.conf** to **gametargets.conf** in the project root

2. Update paths in the **gametargets.conf** for your game. Follow example in the **gametargets.example.conf**. Make sure to remove "#" from the start of all three lines

3. Reconfigure and rebuild each configuration you use so Meson reloads **gametargets.conf**. For example:
    ```powershell
    meson setup --reconfigure _Comp64Release
    meson compile -C _Comp64Release
    ```
    The build deploys binaries to the game directories specified in **gametargets.conf**.

### Profiling Remix
Remix has support for profiling using the [Tracy](https://github.com/wolfpld/tracy) tool, specifically the [v0.8 release](https://github.com/wolfpld/tracy/releases/download/v0.8/Tracy-0.8.7z)

To enable Tracy profiling:
1. Open a command line window in a build folder (i.e. `dxvk-remix/_Comp64Release/`)
2. Run `meson --reconfigure -D enable_tracy=true`
3. Rebuild dxvk-remix-nv

To profile:
1. Launch tracy.exe
2. Launch the game and reach the section you wish to profile
3. When ready, hit `Connect` in Tracy to begin profiling.
4. It's best to collect at least 500 frames worth of data, so you can average out the results.

### Remix API

If there's an intent to use the Remix Renderer in projects with *available* source code, Direct3D 9 API can be utilized, since Remix's `d3d9.dll` implements the Direct3D 9 API.
Alternatively, Remix API can be used to programmatically pass the game data to the Remix Renderer, with *or* instead of Direct3D API. [Click for more info.](/documentation/RemixSDK.md)

## Project Documentation

- [Anti-Culling System](/documentation/AntiCullingSystem.md)
- [Contributing Guide](/CONTRIBUTING.md)
- [Foliage System](/documentation/FoliageSystem.md)
- [GPU Print](/documentation/GpuPrint.md)
- [Opacity Micromap](/documentation/OpacityMicromap.md)
- [Remix API](/documentation/RemixSDK.md)
- [Rtx Options](/RtxOptions.md)
- [Terrain System](/documentation/TerrainSystem.md)
- [Unit Test](/documentation/UnitTest.md)
