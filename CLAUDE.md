# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the DarkSHINE Experiment Trigger project - a Xilinx Vitis HLS implementation of a reduced 3D U-Net for efficient FPGA acceleration. The architecture processes 3D volumetric data (43×43×11) with 1 input channel and produces 5 output channels using a simplified U-Net with a single encoder-decoder level.

## Development Commands

### Weight Conversion
```bash
# Convert PyTorch checkpoints to HLS-compatible weights
python3 convert_weights.py

# Analyze checkpoint contents (debugging)
python3 analyze_checkpoints.py
```

### C++ Alternative Weight Conversion
```bash
# Compile the C++ weight converter
g++ -o checkpoint_converter checkpoint_converter.cpp

# Run the converter
./checkpoint_converter
```

### HLS Development
The project uses Xilinx Vitis HLS with configuration in `hls_component/hls_config.cfg`.
Target FPGA: xczu15eg-ffvb1156-2-i

Key HLS files:
- `hls_component/unet3d_reduced.cpp` - Main implementation
- `hls_component/testbench.cpp` - Verification testbench
- `hls_component/unet_weights.h` - Generated weight headers
- `hls_component/unet_weights_data.cpp` - Generated weight data

## Architecture

### Network Structure
The reduced U-Net implements a single-level encoder-decoder with skip connections:
- **Input**: 1×43×43×11 (single channel 3D volume)
- **Feature Maps**: [64, 128] channels (f_maps[0], f_maps[1])
- **Output**: 5×43×43×11 (five-channel predictions)

### Key Components
1. **Input Convolution Block**: 1→64→64 channels with GroupNorm/ReLU
2. **Encoder Path**: MaxPool 2×2×2 + Conv block (64→128→128)
3. **Decoder Path**: Upsampling + Concatenation + Conv block (192→64→64)
4. **Output Path**: Conv block + final 1×1×1 conv (64→5)

### Weight Management
- PyTorch checkpoints stored in `block_checkpoints/` directory
- Expected checkpoint files: `input_conv.pth`, `encoder_conv.pth`, `decoder_conv.pth`, `output_conv.pth`, `final_conv.pth`
- Weights converted to C++ arrays via `convert_weights.py`
- GroupNorm uses 8 groups throughout the network

## HLS Optimization Details

The implementation uses advanced HLS optimizations:
- **Memory Management**: BRAM, URAM, and LUTRAM for different buffer types
- **Dataflow**: Pipelined processing with streaming interfaces
- **Array Partitioning**: Strategic partitioning for parallel access
- **Fixed-Point**: Uses `ap_fixed.h` for FPGA-optimized arithmetic

## File Dependencies

When modifying weights:
1. Update checkpoints in `block_checkpoints/`
2. Run `convert_weights.py` to regenerate weight files
3. Ensure `unet_weights.h` and `unet_weights_data.cpp` are updated
4. Recompile HLS components

When modifying the network architecture:
1. Update constants in `unet3d_reduced.cpp` (lines 14-28)
2. Adjust memory buffer sizes accordingly
3. Update testbench if input/output dimensions change
