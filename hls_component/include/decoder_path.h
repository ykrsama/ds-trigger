#ifndef DECODER_PATH_H
#define DECODER_PATH_H

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include "model_parameter.h"

void DecoderUpsample3D(
    float input[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void ConcatenateTensors(
    float tensor1[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float tensor2[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void DecoderConv3D_1(
    float kernel[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void DecoderConv3D_2(
    float kernel[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void DecoderGroupNorm3D_1(
    float input_data[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[CONCAT_CHANNELS],
    float beta[CONCAT_CHANNELS],
    float output_data[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void DecoderGroupNorm3D_2(
    float input_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[F_MAP_0],
    float beta[F_MAP_0],
    float output_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void DecoderDoubleConv3D(
    float decoder_conv1_weight[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float decoder_conv1_gamma[CONCAT_CHANNELS],
    float decoder_conv1_beta[CONCAT_CHANNELS],
    float decoder_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float decoder_conv2_gamma[F_MAP_0],
    float decoder_conv2_beta[F_MAP_0],
    float input[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

#endif // DECODER_PATH_H