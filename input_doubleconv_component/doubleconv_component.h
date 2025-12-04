#ifndef DOUBLECONV_COMPONENT_H
#define DOUBLECONV_COMPONENT_H

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include "model_parameter.h"

// Top-level function that only calls DoubleConv3D2Head
void DoubleConvComponent(
    // Input
    float input[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],

    // Weights for DoubleConv3D2Head
    float kernel1[F_MAP_h][IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float gamma1[IN_CHANNELS],
    float beta1[IN_CHANNELS],
    float kernel2[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float gamma2[F_MAP_h],
    float beta2[F_MAP_h],

    // Outputs
    float output1[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output2[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

#endif // DOUBLECONV_COMPONENT_H