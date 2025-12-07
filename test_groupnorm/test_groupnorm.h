#ifndef TEST_GROUPNORM3D_H
#define TEST_GROUPNORM3D_H

#include "../template/model.h"

void TestGroupNorm(
    data_t input_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    data_t gamma[F_MAP_0],
    data_t beta[F_MAP_0],
    data_t output_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

#endif // TEST_GROUPNORM3D_H