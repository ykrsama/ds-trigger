#ifndef TEST_UPSAMPLE3D_H
#define TEST_UPSAMPLE3D_H

#include "../template/model.h"

void TestUpsample(
    data_t input_data[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    data_t output_data[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

#endif // TEST_UPSAMPLE3D_H