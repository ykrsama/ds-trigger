#ifndef TEST_GROUPNORM3D_H
#define TEST_GROUPNORM3D_H

#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include "../template/model.h"

void TestGroupNorm(
    float input_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[F_MAP_0],
    float beta[F_MAP_0],
    float output_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

#endif // TEST_GROUPNORM3D_H