# DarkSHINE Experiment Trigger

This repository contains a Xilinx Vitis HLS implementation of a 3D U-Net with reduced contraction path for efficient FPGA acceleration.

## Architecture Overview

The reduced U-Net architecture follows this design:

```
input(1) → DoubleConv (gcr) → f_maps[0]=64 ────── concat ─────→ f_maps[1]=128 → DoubleConv (gcr) → f_maps[0]=64 → DoubleConv (gcr) → output(5)
                                   │                                  ↑
                                MaxPool                           Upsampling
                                   ↓                                  │
                               f_maps[0]=64 → DoubleConv (gcr) → f_maps[1]=128
```
