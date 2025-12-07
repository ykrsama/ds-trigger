#include "test_upsample.h"

void TestUpsample(
    data_t input_data[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    data_t output_data[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return

    #pragma HLS interface bram port=input_data storage_type=ram_t2p
    #pragma HLS interface bram port=output_data storage_type=ram_t2p

    #pragma HLS dataflow

    Upsample3D<F_MAP_1>(
        input_data, output_data
    );
}