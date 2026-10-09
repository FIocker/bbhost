# DLSS SR and Frame Generation integration

Experimental renderer integration; 3x/4x pacing still needs further validation.

## Sources

- Base: `droogie/bbhost`, commit `8f2746c28611a0255690d012ae53c0e5d9424dbb`.
- Native DLAA foundation: `YoucefNabil/bbhost`, commits `f35fa89` and
  `80d8edf` (the latter tracks sixteen depth targets). The fork's updater
  changes are unrelated and are not included.
- Temporal upscaling and HUD composition reference:
  `Supermedo/bloodborne_pc`, commit `e0761196535bac06e5d5beb1c43da5124a06dd4c`.
- NGX FG resource and camera reference: `AwesomeObserver/bloodborne_pc`,
  commit `c951e0f7876ae50a905c7cbc5e1941a310a99e3d`. The applicable MIT bridge
  notice is preserved in `tools/ngx_bridge/BBPORT-NOTICE.txt`.
- NVIDIA's DLSS and DLSS-FG programming guides are the API reference.

## Validation so far

On an RTX 5080 with driver 617.14:

- NGX reports both SR and Frame Generation available.
- `ngx_probe.exe gpu` reconstructs a 640x360 HDR input into a 960x540 output
  and reads the resulting pixels back from the GPU.
- A moving-object fixture verifies fourteen 2x midpoints, twenty-eight 3x
  subframes and forty-two 4x subframes against their respective temporal
  positions. Each image must fit its expected position better than either
  real endpoint and, for multi-frame generation, another generated position.
  These are GPU pixel checks, not FPS counters.
- A history reset suppresses interpolation, subsequent frames recover, and
  NGX's OutputReal preserves every real input frame exactly.
- An isolated Bloodborne 1.09 run reaches gameplay and evaluates native DLAA
  with camera-derived motion vectors.
- PresentMon 2.6.0 measures 59.99 displayed FPS in the 60 FPS DLAA baseline,
  across 1,797 gameplay intervals. Median display interval: 16.684 ms.
  This baseline does not include Frame Generation.

- The game reconstructs a 1280x720 LDR scene to 1920x1080 before HUD
  composition. A captured gameplay image verifies correct picture extent,
  orientation and HUD placement. NGX logs repeated successful SR evaluations.
- Native DLAA + FG: 115.44 actually displayed FPS over 2,306 intervals.
  SR Quality + FG: 115.43 actually displayed FPS over 4,615 intervals.
  Both have an 8.34 ms median display interval and no undisplayed rows in
  their gameplay windows. The user's external NVIDIA Control Panel cap is
  116 FPS; these measurements are consistent with that cap, not an uncapped
  120 FPS claim. FG reports no evaluation failures; two reset frames in the
  SR run suppress interpolation as expected.
- The packaged executable reaches gameplay with the NGX bridge removed,
  retaining the ordinary renderer. A second packaged run with equal input/output
  dimensions evaluates native DLAA and FG instead of skipping reconstruction.
- The user confirms visible frame generation and correct picture size after
  cropping NGX input to the logical game region. The first implementation
  fed NGX the larger display allocation and showed a small upper-left picture.

The earlier display measurements above describe the original 2x presenter.
The current presenter uses three independently owned output sets, GPU source
timestamps and a separate compositor queue where available. It skips expired
generated positions instead of delaying the next real frame behind them.
Real simulation remains at the selected game cap. A driver cap below the
requested multiplied rate can reduce the number of generated images displayed.

Optional object motion vectors now cover eligible animated meshes as well as
camera movement. Multi-frame generation is experimental and needs supported
RTX hardware. A local 4x functional run with object motion enabled reached
115.31 displayed FPS under a 116 FPS driver cap, with an 8.342 ms median and
16.776 ms p99 display interval. This is a capped functional check, not an
uncapped 4x performance or smoothness claim. NVIDIA Reflex is not integrated.

Core Vulkan validation exposed buffer-device-address flags and conflicting
extension requests, which were corrected. Synchronization validation still
reports clear/copy hazards inside the NVIDIA runtime's evaluations, alongside
existing renderer warnings. It has not passed a clean synchronization run.
Broader hardware testing and controlled pacing comparisons remain necessary.
The clear/copy/layout hazards also reproduce in the standalone NGX fixture,
without bbhost's renderer. NVIDIA tracks matching Vulkan DLSS-G hazards in
[Streamline issue #84](https://github.com/NVIDIA-RTX/Streamline/issues/84).
They are not assumed harmless or suppressed in the validation results.

A 30 FPS source with 3x generation displays about 90 FPS, with zero evaluation
failures, but its display intervals are less even than its submission intervals.
Changing from MAILBOX to IMMEDIATE presentation did not resolve that variation.
These functional runs do not establish smooth foreground gameplay under an
uncapped, otherwise idle GPU. NVIDIA's
[DLSS-G integration guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_G.md)
requires Reflex for its Streamline integration; the native NGX path here does
not supply Streamline's presentation pacer or Reflex automatically.

## Integration checks

- Verified: smaller scene input reconstructs into a larger displayed output,
  with HUD composition at output resolution.
- Depth, motion, camera and HUD-less scene belong to the same real frame.
- Generated and real frames are submitted in temporal order and paced using
  GPU source intervals divided by the selected factor. Expired positions may
  be skipped under backpressure. Driver caps and actual refresh rate limit
  displayed output; generated frames do not increase the simulation rate.
- Inspect camera movement, animated characters, HUD, loading transitions,
  history resets and resolution changes.
- Resource ownership and GPU completion prevent a frame's guides or output
  being overwritten while it is still being evaluated or presented.
- Unsupported hardware, missing DLLs and NGX failures retain normal rendering.

## Tools

The NGX bridge is an independent MSVC CMake project under `tools/ngx_bridge`;
the host remains Clang/mingw-w64. It requires a separately supplied NVIDIA
DLSS SDK and Vulkan headers. `ngx_probe.exe` checks capabilities;
`ngx_probe.exe gpu` runs reconstruction and interpolation/readback checks.

For display cadence, capture a PresentMon CSV with display tracking enabled,
then run `python tools/win/analyze_presentmon.py capture.csv --after-ms 20000`.
Use a gameplay-only window and retain rows for undisplayed frames. Actual
display cadence must be distinguished from submission cadence; cadence alone
does not establish that generated image content is correct.

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

Modes are `off`, `dlaa`, `quality`, `balanced`, `performance`, and
`ultra_performance`. Setup and F10 share the reconstruction selector,
Off/2x/3x/4x frame generation, and object motion setting. Their saved choices take
precedence over config defaults; changes apply after restarting the game.
Unsupported 3x/4x factors fall back to 2x. `BBHOST_DLSS_FG=0`, `2`, `3` or `4`
overrides the selection for a run; legacy `1` still means 2x.

The resolution selector describes native rendering or reconstructed output.
SR presets automatically select a lower, even input size: Quality at 2/3,
Balanced at 0.58, Performance at 1/2 and Ultra Performance at 1/3 of the output
dimensions. Setup's Custom entry accepts arbitrary sizes from 256x144 through
7680x4320 and exposes the saved size as the last F10 resolution choice.
Display buffers and the startup heap include the requested output size.

Legacy explicit `output_width`/`output_height` seed the display selector;
later saved resolution choices still win. `BBHOST_RES` continues to override
the render input for diagnostic/older launchers. Keep that explicit input's
aspect ratio close to the output's. An output at/below scene size, an
incompatible target, or failed SR creation/evaluation retains native DLAA
when NGX is available. Missing NGX retains ordinary game rendering.

The local experimental kit supplies separate SR+FG and DLAA+FG launchers.
It uses the isolated playtest save/config and does not bundle game files or
account credentials. Its in-game FPS counter measures real game frames,
so a reading of 60 does not imply generated presentations are missing.

## Optional object motion vectors

`[dlss] object_motion = true` enables animated-mesh motion tracking. The
F10 graphics menu exposes the same choice; its saved setting takes precedence
over the config default. Restart after changing it. `BBHOST_OBJECT_MOTION=0`
or `1` overrides both for an individual run.

The implementation adapts Supermedo's per-vertex history algorithm from
commit `e076119`. Skinned G-buffer draws store their current clip positions
and load matching positions from the previous frame. A separate RGBA32F
attachment stores motion in pixels, validity, and fragment depth. The depth
check rejects vectors from objects covered by later geometry. Missing history,
camera cuts, and unmatched meshes retain camera motion. The combined vectors
feed both reconstruction and frame generation.

Character skeletons qualify automatically; small skeleton palettes are tracked
when they change, keeping static scenery out of the expensive path. The
position history uses about 128 MiB, plus the motion image and parameter space.
The performance cost depends on the scene and hardware.
