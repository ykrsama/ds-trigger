#ifndef ENCODER_DOUBLECONV_H
#define ENCODER_DOUBLECONV_H

#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include "../template/model.h"

void EncoderDoubleConv(
    data_t input[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    data_t kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    data_t gamma1[F_MAP_0],
    data_t beta1[F_MAP_0],
    data_t kernel2[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    data_t gamma2[F_MAP_0],
    data_t beta2[F_MAP_0],
    data_t output[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
);

#endif // ENCODER_DOUBLECONV_H