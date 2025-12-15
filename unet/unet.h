#ifndef UNET_H
#define UNET_H

#include "../template/model.h"

// Top function declaration
void UNet(data_t input[BATCH_SIZE][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
          data_t conv1_kernel1[F_MAP_h][INPUT_DEPTH][CONV_KERNEL][CONV_KERNEL],
          data_t conv1_kernel2[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL],
          data_t conv2_kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL],
          data_t conv2_kernel2[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL],
          data_t conv3_kernel1[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL],
          data_t conv3_kernel2[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL],
          data_t final_kernel[OUT_CHANNELS][F_MAP_0][1][1][1],
          data_t final_bias[OUT_CHANNELS],
          data_t output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

// 2D Double Convolution template
template<int T_IN_CHANNELS,
         int T_MID_CHANNELS,
         int T_OUT_CHANNELS,
         int T_INPUT_HEIGHT,
         int T_INPUT_WIDTH>
void DoubleConv2D(data_t input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                  data_t kernel1[T_MID_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL],
                  data_t kernel2[T_OUT_CHANNELS][T_MID_CHANNELS][CONV_KERNEL][CONV_KERNEL],
                  data_t output[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

// 2D Concatenation template
template<int T_CHANNELS1, int T_CHANNELS2, int T_CONCAT_CHANNELS, int T_INPUT_HEIGHT, int T_INPUT_WIDTH>
void Concat2D(data_t tensor1[BATCH_SIZE][T_CHANNELS1][T_INPUT_HEIGHT][T_INPUT_WIDTH],
              data_t tensor2[BATCH_SIZE][T_CHANNELS2][T_INPUT_HEIGHT][T_INPUT_WIDTH],
              data_t output[BATCH_SIZE][T_CONCAT_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

// Final 1x1 convolution for output
template<int T_IN_CHANNELS, int T_OUT_CHANNELS, int T_INPUT_HEIGHT, int T_INPUT_WIDTH>
void FinalConv2D(data_t kernel[T_OUT_CHANNELS][T_IN_CHANNELS][1][1][1],
                 data_t bias[T_OUT_CHANNELS],
                 data_t input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                 data_t output[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

// Reshape output from CHW to CDHW format
template<int T_IN_CHANNELS, int T_OUT_CHANNELS, int T_DEPTH, int T_HEIGHT, int T_WIDTH>
void ReshapeOutput(data_t input[BATCH_SIZE][T_IN_CHANNELS][T_HEIGHT][T_WIDTH],
                   data_t output[BATCH_SIZE][T_OUT_CHANNELS][T_DEPTH][T_HEIGHT][T_WIDTH]);

#endif // UNET_H