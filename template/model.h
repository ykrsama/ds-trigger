#ifndef MODEL_H
#define MODEL_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include "../template/model_parameter.h"

typedef ap_fixed<32,16> data_t;

// Template function declarations
template<int T_IN_CHANNELS,
         int T_INPUT_DEPTH = INPUT_DEPTH,
         int T_INPUT_HEIGHT = INPUT_HEIGHT,
         int T_INPUT_WIDTH = INPUT_WIDTH>
void GroupNorm3D(data_t input_data[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                 data_t gamma[T_IN_CHANNELS],
                 data_t beta[T_IN_CHANNELS],
                 data_t output_data[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

template<int T_IN_CHANNELS,
         int T_OUT_CHANNELS,
         int T_INPUT_DEPTH = INPUT_DEPTH,
         int T_INPUT_HEIGHT = INPUT_HEIGHT,
         int T_INPUT_WIDTH = INPUT_WIDTH>
void Conv3D(data_t kernel[T_OUT_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
            data_t input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
            data_t output[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

template<int T_IN_CHANNELS,
         int T_MID_CHANNELS,
         int T_OUT_CHANNELS,
         int T_INPUT_DEPTH = INPUT_DEPTH,
         int T_INPUT_HEIGHT = INPUT_HEIGHT,
         int T_INPUT_WIDTH = INPUT_WIDTH>
void DoubleConv3D(data_t input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                  data_t kernel1[T_MID_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
//                  data_t gamma1[T_IN_CHANNELS],
//                  data_t beta1[T_IN_CHANNELS],
                  data_t kernel2[T_OUT_CHANNELS][T_MID_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
//                  data_t gamma2[T_MID_CHANNELS],
//                  data_t beta2[T_MID_CHANNELS],
                  data_t output[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

template<int T_IN_CHANNELS,
    int T_MID_CHANNELS,
    int T_OUT_CHANNELS,
    int T_INPUT_DEPTH = INPUT_DEPTH,
    int T_INPUT_HEIGHT = INPUT_HEIGHT,
    int T_INPUT_WIDTH = INPUT_WIDTH>
void DoubleConv3D2Head(data_t input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                       data_t kernel1[T_MID_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
//                       data_t gamma1[T_IN_CHANNELS],
//                       data_t beta1[T_IN_CHANNELS],
                       data_t kernel2[T_OUT_CHANNELS][T_MID_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
//                       data_t gamma2[T_MID_CHANNELS],
//                       data_t beta2[T_MID_CHANNELS],
                       data_t output1[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                       data_t output2[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

template<int T_IN_CHANNELS>
void MaxPool3D(data_t input[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
               data_t output[BATCH_SIZE][T_IN_CHANNELS][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]);

template<int T_IN_CHANNELS>
void Upsample3D(data_t input[BATCH_SIZE][T_IN_CHANNELS][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
                data_t output[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

template<int T_CHANNELS1, int T_CHANNELS2, int T_CONCAT_CHANNELS>
void ConcatenateTensors(data_t tensor1[BATCH_SIZE][T_CHANNELS1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                        data_t tensor2[BATCH_SIZE][T_CHANNELS2][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                        data_t output[BATCH_SIZE][T_CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

template<int T_IN_CHANNELS, int T_OUT_CHANNELS>
void FinalConv1x1(data_t kernel[T_OUT_CHANNELS][T_IN_CHANNELS][1][1][1],
                  data_t bias[T_OUT_CHANNELS],
                  data_t input[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                  data_t output[BATCH_SIZE][T_OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

void Sigmoid3D(data_t input[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
               data_t output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

// Conv2D
template<int T_IN_CHANNELS,
    int T_OUT_CHANNELS,
    int T_INPUT_HEIGHT,
    int T_INPUT_WIDTH>
void Conv2D(
    data_t kernel[T_OUT_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL],
    data_t input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH],
    data_t output[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

// DoubleConv2D2Head
template<int T_IN_CHANNELS,
    int T_MID_CHANNELS,
    int T_OUT_CHANNELS,
    int T_INPUT_HEIGHT,
    int T_INPUT_WIDTH>
void DoubleConv2D2Head(data_t input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                       data_t kernel1[T_MID_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL],
                       data_t kernel2[T_OUT_CHANNELS][T_MID_CHANNELS][CONV_KERNEL][CONV_KERNEL],
                       data_t output1[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                       data_t output2[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

// MaxPool2D
template<int T_IN_CHANNELS,
         int T_INPUT_HEIGHT,
         int T_INPUT_WIDTH>
void MaxPool2D(data_t input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH],
               data_t output[BATCH_SIZE][T_IN_CHANNELS][(T_INPUT_HEIGHT/POOL_STRIDE)][(T_INPUT_WIDTH/POOL_STRIDE)]);

#endif // MODEL_H