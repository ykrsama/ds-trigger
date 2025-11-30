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

using namespace std;

#define BATCH_SIZE 1
#define INPUT_CHANNELS 1
#define F_MAPS_0 64
#define F_MAPS_1 128
#define OUTPUT_CHANNELS 5
#define INPUT_DEPTH 43
#define INPUT_HEIGHT 43
#define INPUT_WIDTH 11
#define KERNEL_SIZE 3
#define PADDING 1
#define STRIDE 1
#define POOL_SIZE 2
#define POOL_STRIDE 2
#define NUM_GROUPS 8
#define EPSILON 1e-5f
#define HALF_DEPTH (INPUT_DEPTH/2)
#define HALF_HEIGHT (INPUT_HEIGHT/2)
#define HALF_WIDTH (INPUT_WIDTH/2)

// Function declarations
void InputConv3dBlock(
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float conv1_kernel[F_MAPS_0][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv1_gamma[F_MAPS_0],
    float conv1_beta[F_MAPS_0],
    float conv2_kernel[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv2_gamma[F_MAPS_0],
    float conv2_beta[F_MAPS_0],
    float output[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void EncoderConv3dBlock(
    float input[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH/2][INPUT_HEIGHT/2][INPUT_WIDTH/2],
    float conv1_kernel[F_MAPS_1][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv1_gamma[F_MAPS_1],
    float conv1_beta[F_MAPS_1],
    float conv2_kernel[F_MAPS_1][F_MAPS_1][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv2_gamma[F_MAPS_1],
    float conv2_beta[F_MAPS_1],
    float output[BATCH_SIZE][F_MAPS_1][INPUT_DEPTH/2][INPUT_HEIGHT/2][INPUT_WIDTH/2]
);

void DecoderConv3dBlock(
    float input[BATCH_SIZE][F_MAPS_0+F_MAPS_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float conv1_kernel[F_MAPS_0][F_MAPS_0+F_MAPS_1][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv1_gamma[F_MAPS_0],
    float conv1_beta[F_MAPS_0],
    float conv2_kernel[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv2_gamma[F_MAPS_0],
    float conv2_beta[F_MAPS_0],
    float output[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void OutputConv3dBlock(
    float input[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float conv1_kernel[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv1_gamma[F_MAPS_0],
    float conv1_beta[F_MAPS_0],
    float conv2_kernel[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv2_gamma[F_MAPS_0],
    float conv2_beta[F_MAPS_0],
    float output[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void MaxPool3d(
    float input[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH/2][INPUT_HEIGHT/2][INPUT_WIDTH/2]
);

void NearestUpsample3d(
    float input[BATCH_SIZE][F_MAPS_1][INPUT_DEPTH/2][INPUT_HEIGHT/2][INPUT_WIDTH/2],
    float output[BATCH_SIZE][F_MAPS_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void Concatenate3d(
    float input1[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float input2[BATCH_SIZE][F_MAPS_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAPS_0+F_MAPS_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void unet3d_reduced(
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float input_conv1_weight[F_MAPS_0][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float input_conv1_gamma[F_MAPS_0],
    float input_conv1_beta[F_MAPS_0],
    float input_conv2_weight[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float input_conv2_gamma[F_MAPS_0],
    float input_conv2_beta[F_MAPS_0],
    float encoder_conv1_weight[F_MAPS_1][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float encoder_conv1_gamma[F_MAPS_1],
    float encoder_conv1_beta[F_MAPS_1],
    float encoder_conv2_weight[F_MAPS_1][F_MAPS_1][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float encoder_conv2_gamma[F_MAPS_1],
    float encoder_conv2_beta[F_MAPS_1],
    float decoder_conv1_weight[F_MAPS_0][F_MAPS_0+F_MAPS_1][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float decoder_conv1_gamma[F_MAPS_0],
    float decoder_conv1_beta[F_MAPS_0],
    float decoder_conv2_weight[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float decoder_conv2_gamma[F_MAPS_0],
    float decoder_conv2_beta[F_MAPS_0],
    float output_conv1_weight[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float output_conv1_gamma[F_MAPS_0],
    float output_conv1_beta[F_MAPS_0],
    float output_conv2_weight[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float output_conv2_gamma[F_MAPS_0],
    float output_conv2_beta[F_MAPS_0],
    float final_conv_weight[OUTPUT_CHANNELS][F_MAPS_0][1][1][1],
    float final_conv_bias[OUTPUT_CHANNELS],
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

#endif // UNET3D_REDUCED_H