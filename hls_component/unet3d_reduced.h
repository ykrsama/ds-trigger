#ifndef UNET3D_REDUCED_H
#define UNET3D_REDUCED_H

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "unet_weights.h"

using namespace std;

// Network configuration constants
// Default configuration for UNet3DFPGAModular with f_maps=[64, 128]
#define BATCH_SIZE 1
#define INPUT_CHANNELS 1
#define F_MAP_0 64
#define F_MAP_1 128

// Input dimensions (configurable based on dataset)
#define INPUT_DEPTH 43
#define INPUT_HEIGHT 43
#define INPUT_WIDTH 11
#define OUTPUT_CHANNELS 5

// Convolution parameters
#define CONV_KERNEL 3
#define CONV_PADDING 1
#define CONV_STRIDE 1

// Pooling parameters
#define POOL_KERNEL 2
#define POOL_STRIDE 2

// Calculated dimensions
#define PADDED_DEPTH (INPUT_DEPTH + 2 * CONV_PADDING)
#define PADDED_HEIGHT (INPUT_HEIGHT + 2 * CONV_PADDING)
#define PADDED_WIDTH (INPUT_WIDTH + 2 * CONV_PADDING)

#define CONV_OUTPUT_DEPTH ((INPUT_DEPTH + 2 * CONV_PADDING - CONV_KERNEL) / CONV_STRIDE + 1)
#define CONV_OUTPUT_HEIGHT ((INPUT_HEIGHT + 2 * CONV_PADDING - CONV_KERNEL) / CONV_STRIDE + 1)
#define CONV_OUTPUT_WIDTH ((INPUT_WIDTH + 2 * CONV_PADDING - CONV_KERNEL) / CONV_STRIDE + 1)

#define POOL_OUTPUT_DEPTH (CONV_OUTPUT_DEPTH / POOL_STRIDE)
#define POOL_OUTPUT_HEIGHT (CONV_OUTPUT_HEIGHT / POOL_STRIDE)
#define POOL_OUTPUT_WIDTH (CONV_OUTPUT_WIDTH / POOL_STRIDE)

#define UPSAMPLE_OUTPUT_DEPTH (POOL_OUTPUT_DEPTH * 2)
#define UPSAMPLE_OUTPUT_HEIGHT (POOL_OUTPUT_HEIGHT * 2)
#define UPSAMPLE_OUTPUT_WIDTH (POOL_OUTPUT_WIDTH * 2)

#define CONCAT_CHANNELS (F_MAP_0 + F_MAP_1)
#define FMAP_h (F_MAP_0 / 2)  // Half of F_MAP_0 for input conv block

// Function declarations for individual blocks

// Input convolution block: INPUT_CHANNELS -> F_MAP_0
void InputConv3D(
    float kernel1[F_MAP_0][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float kernel2[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

// MaxPool3D block: spatial downsampling by 2
void EncoderMaxPool3D(
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
);

// Encoder convolution block: F_MAP_0 -> F_MAP_1
void EncoderConv3D(
    float kernel1[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float kernel2[F_MAP_1][F_MAP_1][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
);

// Nearest neighbor upsampling: spatial upsampling by 2
void DecoderUpsample3D(
    float input[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

// Decoder convolution block: CONCAT_CHANNELS -> F_MAP_0
void DecoderConv3D(
    float kernel1[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float kernel2[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

// Output convolution block: F_MAP_0 -> F_MAP_0
void OutputConv3D(
    float kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float kernel2[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

// Final 1x1 convolution: F_MAP_0 -> OUTPUT_CHANNELS
void FinalConv1x1(
    float kernel[OUTPUT_CHANNELS][F_MAP_0][1][1][1],
    float bias[OUTPUT_CHANNELS],
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

// Utility functions for concatenation and activation
void ConcatenateTensors(
    float tensor1[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float tensor2[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void Sigmoid3D(
    float input[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

// Top-level UNet3D function
void UNet3DReduced(
    // Input
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],

    // Weights for all layers
    float input_conv1_weight[F_MAP_h][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input_conv2_weight[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float encoder_conv1_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float encoder_conv2_weight[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float decoder_conv1_weight[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float decoder_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float output_conv1_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float output_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float final_conv_weight[OUTPUT_CHANNELS][F_MAP_0][1][1][1],
    float final_conv_bias[OUTPUT_CHANNELS],

    // Output
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

#endif // UNET3D_REDUCED_H