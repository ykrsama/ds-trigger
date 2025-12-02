#ifndef INPUT_PATH_H
#define INPUT_PATH_H

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include "model_parameter.h"

void InputConv3D_1(
    float kernel[F_MAP_h][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void InputConv3D_2(
    float kernel[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void InputGroupNorm3D_1(
    float input_data[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[INPUT_CHANNELS],
    float beta[INPUT_CHANNELS],
    float output_data[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void InputGroupNorm3D_2(
    float input_data[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[F_MAP_h],
    float beta[F_MAP_h],
    float output_data[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void InputDoubleConv3D(
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float kernel1[F_MAP_h][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float gamma1[INPUT_CHANNELS],
    float beta1[INPUT_CHANNELS],
    float kernel2[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float gamma2[F_MAP_h],
    float beta2[F_MAP_h],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);
#endif // INPUT_PATH_H