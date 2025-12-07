#ifndef TEST_CONV3D_H
#define TEST_CONV3D_H

#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include "../template/model.h"

void  TestConv(
    data_t kernel[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    data_t input[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    data_t output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

#endif // TEST_CONV3D_H