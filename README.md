# DarkSHINE Experiment Trigger

This repository contains a Xilinx Vitis HLS implementation of a 3D U-Net with reduced contraction path for efficient FPGA acceleration.

## Architecture Overview

The reduced U-Net architecture follows this design:

```
input(1) → Conv+ReLU → f_maps[0]=64 ─── concat ──→ f_maps[1]=128 → Conv+ReLU → f_maps[0]=64 → Conv+ReLU → output(5)
                          │                         ↑
                       MaxPool                  Upsampling
                          ↓                         │
                       f_maps[0]=64 → Conv+ReLU → f_maps[1]=128
```

**Key Features:**
- **Input**: 1 channel, dimensions [43×43×11]
- **Output**: 5 channels, same spatial dimensions
- **Feature maps**: [64, 128] channels (f_maps[0], f_maps[1])
- **Single encoder-decoder level** for reduced complexity
- **Skip connections** via concatenation
- **GroupNorm** with 8 groups for normalization
- **ReLU activation** throughout the network

## File Structure

### Core Implementation

### Weight Conversion
- `convert_weights.py` - PyTorch checkpoint to HLS weight converter
- `checkpoint_converter.cpp` - C++ alternative weight converter
- `analyze_checkpoints.py` - Checkpoint analysis utility
- `hls_component/unet_weights.h` - Generated weight declarations
- `hls_component/unet_weights_data.cpp` - Generated weight data arrays
- `block_checkpoints/` - Directory containing PyTorch checkpoints

## Architecture Details

### Network Blocks

1. **Input Convolution Block** (1 → 64 → 64 channels)
   - Two 3×3×3 convolutions with GroupNorm and ReLU
   - Produces feature maps for skip connection

2. **Encoder Path**
   - MaxPool 2×2×2 downsampling
   - Convolution block (64 → 128 → 128 channels)

3. **Decoder Path**
   - Nearest neighbor 2× upsampling
   - Concatenation with skip connection (64 + 128 = 192 channels)
   - Convolution block (192 → 64 → 64 channels)

4. **Output Path**
   - Output convolution block (64 → 64 → 64 channels)
   - Final 1×1×1 convolution (64 → 5 channels)
