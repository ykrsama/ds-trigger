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
#include "unet_weights.h"
#include "input_pth.h"
#include "encoder_path.h"
#include "decoder_path.h"
#include "output_path.h"

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

    // GroupNorm parameters
    float input_conv1_gamma[INPUT_CHANNELS],
    float input_conv1_beta[INPUT_CHANNELS],
    float input_conv2_gamma[F_MAP_h],
    float input_conv2_beta[F_MAP_h],

    float encoder_conv1_gamma[F_MAP_0],
    float encoder_conv1_beta[F_MAP_0],
    float encoder_conv2_gamma[F_MAP_0],
    float encoder_conv2_beta[F_MAP_0],

    float decoder_conv1_gamma[CONCAT_CHANNELS],
    float decoder_conv1_beta[CONCAT_CHANNELS],
    float decoder_conv2_gamma[F_MAP_0],
    float decoder_conv2_beta[F_MAP_0],

    float output_conv1_gamma[F_MAP_0],
    float output_conv1_beta[F_MAP_0],
    float output_conv2_gamma[F_MAP_0],
    float output_conv2_beta[F_MAP_0],

    // Output
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

#endif // UNET3D_REDUCED_H