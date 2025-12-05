#ifndef MODEL_H
#define MODEL_H

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include "../template/model_parameter.h"

// Template function declarations
template<int T_IN_CHANNELS,
         int T_INPUT_DEPTH = INPUT_DEPTH,
         int T_INPUT_HEIGHT = INPUT_HEIGHT,
         int T_INPUT_WIDTH = INPUT_WIDTH>
void GroupNorm3D(float input_data[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                 float gamma[T_IN_CHANNELS],
                 float beta[T_IN_CHANNELS],
                 float output_data[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

template<int T_IN_CHANNELS,
         int T_OUT_CHANNELS,
         int T_INPUT_DEPTH = INPUT_DEPTH,
         int T_INPUT_HEIGHT = INPUT_HEIGHT,
         int T_INPUT_WIDTH = INPUT_WIDTH>
void Conv3D(float kernel[T_OUT_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
            float input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
            float output[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

template<int T_IN_CHANNELS,
         int T_MID_CHANNELS,
         int T_OUT_CHANNELS,
         int T_INPUT_DEPTH = INPUT_DEPTH,
         int T_INPUT_HEIGHT = INPUT_HEIGHT,
         int T_INPUT_WIDTH = INPUT_WIDTH>
void DoubleConv3D(float input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                  float kernel1[T_MID_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
                  float gamma1[T_IN_CHANNELS],
                  float beta1[T_IN_CHANNELS],
                  float kernel2[T_OUT_CHANNELS][T_MID_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
                  float gamma2[T_MID_CHANNELS],
                  float beta2[T_MID_CHANNELS],
                  float output[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

template<int T_IN_CHANNELS,
    int T_MID_CHANNELS,
    int T_OUT_CHANNELS,
    int T_INPUT_DEPTH = INPUT_DEPTH,
    int T_INPUT_HEIGHT = INPUT_HEIGHT,
    int T_INPUT_WIDTH = INPUT_WIDTH>
void DoubleConv3D2Head(float input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                       float kernel1[T_MID_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
                       float gamma1[T_IN_CHANNELS],
                       float beta1[T_IN_CHANNELS],
                       float kernel2[T_OUT_CHANNELS][T_MID_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
                       float gamma2[T_MID_CHANNELS],
                       float beta2[T_MID_CHANNELS],
                       float output1[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                       float output2[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH]);

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
void FinalConv1x1(float kernel[T_OUT_CHANNELS][T_IN_CHANNELS][1][1][1],
                  float bias[T_OUT_CHANNELS],
                  float input[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                  float output[BATCH_SIZE][T_OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

void Sigmoid3D(float input[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
               float output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);


#endif // MODEL_H