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
#include "model_parameter.h"

// Template function declarations
template<int T_IN_CHANNELS>
void GroupNorm3D(float input_data[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                 float gamma[T_IN_CHANNELS],
                 float beta[T_IN_CHANNELS],
                 float output_data[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

template<int T_IN_CHANNELS, int T_OUT_CHANNELS>
void Conv3d(float kernel[T_OUT_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
            float input[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
            float output[BATCH_SIZE][T_OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

template<int T_IN_CHANNELS, int T_MID_CHANNELS, int T_OUT_CHANNELS>
void DoubleConv3D(float input[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                  float kernel1[T_MID_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
                  float gamma1[T_IN_CHANNELS],
                  float beta1[T_IN_CHANNELS],
                  float kernel2[T_OUT_CHANNELS][T_MID_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
                  float gamma2[T_MID_CHANNELS],
                  float beta2[T_MID_CHANNELS],
                  float output[BATCH_SIZE][T_OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

template<int T_IN_CHANNELS, int T_MID_CHANNELS, int T_OUT_CHANNELS>
void DoubleConv3D2Head(float input[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                       float kernel1[T_MID_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
                       float gamma1[T_IN_CHANNELS],
                       float beta1[T_IN_CHANNELS],
                       float kernel2[T_OUT_CHANNELS][T_MID_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
                       float gamma2[T_MID_CHANNELS],
                       float beta2[T_MID_CHANNELS],
                       float output1[BATCH_SIZE][T_OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                       float output2[BATCH_SIZE][T_OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

template<int T_IN_CHANNELS>
void MaxPool3D(float input[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
               float output[BATCH_SIZE][T_IN_CHANNELS][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]);

template<int T_IN_CHANNELS>
void Upsample3D(float input[BATCH_SIZE][T_IN_CHANNELS][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
                float output[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

template<int T_CHANNELS1, int T_CHANNELS2, int T_CONCAT_CHANNELS>
void ConcatenateTensors(float tensor1[BATCH_SIZE][T_CHANNELS1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                        float tensor2[BATCH_SIZE][T_CHANNELS2][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                        float output[BATCH_SIZE][T_CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

template<int T_IN_CHANNELS, int T_OUT_CHANNELS>
void FinalConv1x1(float input[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                  float kernel[T_OUT_CHANNELS][T_IN_CHANNELS][1][1][1],
                  float bias[T_OUT_CHANNELS],
                  float output[BATCH_SIZE][T_OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

void Sigmoid3D(float input[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
               float output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

// Top-level UNet3D function
void UNet3DReduced(
    // Input
    float input[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],

    // Weights for all layers
    float input_conv1_weight[F_MAP_h][IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input_conv2_weight[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float encoder_conv1_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float encoder_conv2_weight[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float decoder_conv1_weight[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float decoder_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float output_conv1_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float output_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float final_conv_weight[OUT_CHANNELS][F_MAP_0][1][1][1],
    float final_conv_bias[OUT_CHANNELS],

    // GroupNorm parameters
    float input_conv1_gamma[IN_CHANNELS],
    float input_conv1_beta[IN_CHANNELS],
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
    float output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

#endif // UNET3D_REDUCED_H