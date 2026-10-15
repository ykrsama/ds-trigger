# UNet3DFPGA weights

Generated from the checkpoint recorded in `weights_manifest.json`.
Compile `unet_weights_data.cpp` with your HLS sources and include `unet_weights.h`.
All arrays are in namespace `unet_fpga_weights`; kernels use PyTorch OIHW order,
with no spatial flipping. The final kernel has shape [classes*depth][features][1][1][1]
to match the existing HLS FinalConv2D template. It contains the same values as
PyTorch's [classes*depth][features][1][1] tensor.

Default type: float32. For HLS ap_fixed<32,16>, compile ALL translation units
with `-DUNET_FPGA_FIXED_POINT`. An alternative type can be selected using
`UNET_FPGA_WEIGHT_TYPE` (its type declaration must be visible in all units).
Fixed-point conversion quantizes parameters; arithmetic accuracy must be tested.

For this model: input NCDHW=(1,1,4,5,5), f_maps=[16,32], five output classes.
Final output channel index = class_index*4 + depth_index. Input depth is folded
into Conv2D channels. Pooled HDF5 data use WHD order: transpose (2,1,0) before input.

These files contain parameters only. The existing unet implementation needs:
- 4 input depth channels, height/width 5, feature channels 16 and 32;
- all six intermediate convolution bias arrays applied before ReLU;
- final convolution with 20 output channels and its bias;
- nearest interpolation directly from 2x2 to 5x5, matching PyTorch;
- per-voxel softmax over the five classes after reshape (for probabilities).
There are no GroupNorm parameters. Existing HLS inference code is not modified.
