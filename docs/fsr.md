# AMD FidelityFX Super Resolution (FSR) Upscaling & FG

bbhost includes independent Vulkan adapters for AMD FidelityFX Super Resolution:
- **FSR 3.1.5 upscaler**: analytical temporal upscaling (Native AA, Quality, Balanced, Performance, Ultra Performance).
- **FSR 3.1.6 frame generation**: analytical optical flow and frame interpolation.
- **FSR 4 (source-v07)**: experimental INT8/DOT4 neural upscaling model.

## Requirements

### FSR 3.1.5 & FSR 3.1.6
- Vulkan 1.3 capable GPU.
- Support for 16-bit storage / float16.

### FSR 4 (source-v07)
- Vulkan 1.3 with:
  - `shaderFloat16` and `shaderInt8` (Vulkan 1.2)
  - `shaderInt16` (Vulkan 1.0)
  - `shaderIntegerDotProduct` (Vulkan 1.3)
  - `shaderStorageImageExtendedFormats` (Vulkan 1.0)
- Neural network model shaders & weights installed in `fsr4_shaders/` (or specified by `BBHOST_FSR4_ASSETS` / `BB_FSR4_DIR`).
- If hardware features or asset files are unavailable, FSR 4 automatically logs a message once and falls back cleanly to FSR 3.1.5.

## Installing FSR 4 Assets on Windows

Run the installation script to download and verify the model bundle:

```powershell
.\tools\win\Install-FSR4.ps1
```

or via the command prompt helper:

```cmd
tools\win\Install-FSR4.bat
```

The script downloads the MIT-licensed FSR 4 v07 INT8/DOT4 shader assets from the upstream Q2RTX repository and verifies file sizes and SHA-256 hashes against the manifest.
