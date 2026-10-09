# DLSS reconstruction and frame generation

Optional native DLAA, DLSS Super Resolution, 2x frame generation and animated
object motion vectors. Temporal settings require a restart. Normal Setup and
F10 menus offer Off/2x generation; higher factors are opt-in experiments.

## Sources

- Base: `droogie/bbhost`, commit `8f2746c28611a0255690d012ae53c0e5d9424dbb`.
- Native DLAA foundation: `YoucefNabil/bbhost`, commits `f35fa89` and
  `80d8edf` (sixteen depth targets). Unrelated updater changes are excluded.
- Upscaling, HUD composition and per-vertex history reference:
  `Supermedo/bloodborne_pc`, commit `e0761196535bac06e5d5beb1c43da5124a06dd4c`.
- NGX FG resources and camera reference: `AwesomeObserver/bloodborne_pc`,
  commit `c951e0f7876ae50a905c7cbc5e1941a310a99e3d`. The adapted MIT bridge
  notice is in `tools/ngx_bridge/BBPORT-NOTICE.txt`.
- NVIDIA's DLSS and DLSS-FG SDK programming guides are the API reference.

## Pipeline and pacing

The lower-resolution scene reconstructs before the game's output-resolution
HUD pass. Host overlays and menu pointer coordinates use that output space.
NGX input is cropped to the logical picture rather than including unused
rows and columns in the larger display allocation.

Three independently owned output sets separate evaluation from presentation.
The producer submits NGX on the renderer queue; the compositor uses a separate
queue where available. Leases last through the final compositor fence,
keeping inputs and outputs alive until their consumers finish. Missing or
unmatched guides retain ordinary presentation.

GPU timestamps estimate source cadence. A filtered phase rejects isolated
completion/evaluation jitter and follows sustained rendering-load changes.
The simulation floor applies after cadence estimation: clamping each sample
first biases zero-mean jitter upward and can send presentation progressively
into the future. Phase error is bounded to half a source period so output-set
drops cannot amplify that delay.

The compositor finishes its GPU copy before sleeping for the presentation
deadline, so copy/overlay cost does not move every scheduled present later.
Driver-blocking time is measured separately from intentional pacing. Expired
generated positions are skipped instead of delaying a real image behind every
interpolated image. Render-anchor IDs reset history across gaps. Generated
output does not increase simulation speed.

## Validation

The Windows Clang/mingw-w64 host and independent MSVC NGX bridge build locally.
Renderer tests cover presets, output/pointer coordinates, history/deadlines,
vertex history, generated SPIR-V and motion-vector GPU readback. A timing test
reproduces capped-jitter phase drift and fails without its fix. Linux CI also
builds and runs registered tests; hosted GPU/game-file skips do not replace
local gameplay evidence.

On RTX 5080, driver 617.14, the standalone NGX GPU fixture checks SR pixels,
fourteen 2x midpoints, twenty-eight 3x thirds and forty-two 4x quarters against
their expected temporal positions. It checks reset recovery and exact OutputReal
preservation. These are pixel checks, not FPS counters.

PresentMon 2.6.0 comparisons use foreground gameplay on an otherwise idle GPU,
with the external NVIDIA Control Panel limit unchanged at 116 FPS. Same-save
runs exclude the first 60 seconds for startup/warmup and the last shutdown
second. The presentation surface is 1920x1080; GPU-bound native DLAA requests
about 7.0 megapixels internally. Object motion is enabled. Actual display
intervals, including undisplayed rows, are inspected separately from submission
intervals and the in-game FPS counter.

| Test | Real source | Displayed FPS | p99 display interval |
| --- | ---: | ---: | ---: |
| Previous pacing, 2x | 30 FPS | 60.0 | 22.10 / 22.39 ms |
| GPU-completed pacing, 2x, two repeats | 30 FPS | 60.0 | 17.76 / 17.71 ms |
| Final alternating A/B, previous / revised timing | 30 FPS | 60.0 / 60.0 | 19.34 / 17.90 ms |
| GPU-bound native DLAA, previous pacing | about 43 FPS | 85.31 | 19.31 ms |
| GPU-bound native DLAA, revised pacing | about 43 FPS | 84.98 / 85.87 | 15.04 / 14.92 ms |
| GPU-bound native DLAA, final phase bound | about 43 FPS | 85.28 | 15.67 ms |
| FG off, same native workload | about 48 FPS | 48.03 | 31.24 ms |
| Hunter's Dream, moving camera, 2x | 30 FPS | 59.97 | 18.63 ms |
| Hunter's Dream, bounded phase, moving camera, 2x | about 58 FPS | 112.67 | 16.57 ms |

The initial Dream camera sweep exposed phase runaway: completion jitter biased
the old filter, output sets backed up and display rate fell to 14-40 FPS.
The unbiased, bounded phase fix removes the sustained collapse in the repeated
sweep. Ordinary loading/shader hitches can still occur; FG cannot supply new
source images through a loading gap.

### Experimental higher factors

3x at 30 source FPS reached 90 displayed FPS with a 13.44 ms p99 interval.
4x at 30 FPS reached 115.72 displayed FPS under the driver cap, with less even
spacing. At 60 source FPS, 3x reached 110.36 displayed FPS with a 20.54 ms p99
interval and cadence problems. Higher factors stay behind
`BBHOST_EXPERIMENTAL_MFG=1`, not normal validated choices. These measurements
precede the final phase bound and do not establish cap-safe MFG.
Unsupported higher factors fall back to 2x.

After the final phase fix, a clean 3x repeat with the game cap at 60 FPS
measured about 55 real FPS, 115.09 displayed FPS and a 16.47 ms p99 interval.
It improved over the earlier run but still varied from 4.95 ms at p10 to
13.05 ms at p90, skipping many generated positions against the driver limit.
This does not justify promoting it to the normal menu.

Reflex is not integrated. NVIDIA's
[Streamline DLSS-G guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_G.md)
requires Reflex for that integration; native NGX does not supply Streamline's
presentation pacer or Reflex automatically.

### Validation limitations

Synchronization validation reports NVIDIA runtime clear/copy/layout hazards,
also reproduced in the standalone fixture without bbhost's renderer, plus
existing renderer warnings. Matching NVIDIA Vulkan DLSS-G hazards are tracked
in [Streamline issue #84](https://github.com/NVIDIA-RTX/Streamline/issues/84).
They are recorded, not suppressed; clean synchronization validation is not
claimed. Broader hardware, window-mode and long-session testing remain necessary.
An oversized stress test exhausted the game's direct-memory allocation and
produced no generated presentations; it is excluded from FG results.
This is not a Reflex/latency certification.

## Building and diagnostics

`tools/ngx_bridge` is an independent MSVC CMake project. The host remains
Clang/mingw-w64. Supply NVIDIA's DLSS SDK and Vulkan headers separately;
SDK/runtime binaries and game files are not included in the source.
`ngx_probe.exe` checks capabilities; `ngx_probe.exe gpu` checks reconstruction,
interpolation and GPU readback.

Capture PresentMon CSVs with display tracking, then run
`python tools/win/analyze_presentmon.py capture.csv --after-ms 60000`.
Choose a gameplay-only interval and retain undisplayed rows. Distinct temporal
pixels and actual display cadence need separate verification.

`BBHOST_FG_SMOOTH=0` and `BBHOST_FG_LATE_PACE=0` restore the previous timing
for A/B comparisons. `BBHOST_FG_TIMING=1` logs GPU anchors, period,
shown/expired positions and driver-blocking time. These are diagnostics.

## Configuration

```toml
[video]
resolution = "1920x1080"
fps_cap = 60

[dlss]
mode = "quality"
output_width = 0
output_height = 0
frame_generation = true
frame_generation_factor = 2
object_motion = true
```

Modes: `off`, `dlaa`, `quality`, `balanced`, `performance`, `ultra_performance`.
Setup and F10 share reconstruction, Off/2x generation, object motion and
resolution choices. Saved choices take precedence over config defaults.
Restart after changing temporal settings. Without the experimental opt-in,
previously saved 3x/4x choices safely select 2x. `BBHOST_DLSS_FG=0` or `2`
overrides a run; legacy `1` means 2x. With `BBHOST_EXPERIMENTAL_MFG=1`,
menus and overrides also allow `3` or `4` on supported hardware.
Keep simulation at 60 FPS for multiplayer compatibility.

The resolution selector describes native rendering or reconstructed output.
Quality uses 2/3 of each dimension, Balanced 0.58, Performance 1/2 and Ultra
Performance 1/3. Input dimensions are even. Setup's Custom entry accepts
256x144 through 7680x4320 and adds the saved size to F10. Display allocations
and the startup heap account for requested output dimensions.

Legacy explicit `output_width`/`output_height` seed the selector; later saved
choices still win. `BBHOST_RES` remains an input-size diagnostic override.
Keep its aspect ratio close to the output. Failed SR can retain native DLAA
when NGX is available; missing NGX retains ordinary rendering.
The host FPS counter counts presentation requests, including generated ones;
PresentMon checks which images actually reach the display.

## Optional object motion vectors

`[dlss] object_motion = true` enables animated-mesh tracking. Setup and F10
expose the same restart-required choice. `BBHOST_OBJECT_MOTION=0` or `1`
overrides it for a run. Combined vectors feed both SR and FG.

Skinned G-buffer draws store current clip positions and load matched previous
positions. A separate RGBA32F attachment stores motion in pixels, validity and
fragment depth. Its depth check rejects vectors covered by later geometry.
Missing history, camera cuts and unmatched meshes retain camera motion.
Character skeletons qualify automatically; small changing skeleton palettes
are tracked, leaving static scenery out of the expensive path.

Position history uses about 128 MiB, plus the motion image and parameters.
Cost depends on scene and hardware. The pacing fix adds no extra object-motion
pass or GPU allocation.
