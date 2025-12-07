#include "test_groupnorm.h"

void TestGroupNorm(
    float input_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[F_MAP_0],
    float beta[F_MAP_0],
    float output_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return

    #pragma HLS interface bram port=input_data storage_type=ram_t2p
    #pragma HLS interface bram port=gamma storage_type=ram_t2p
    #pragma HLS interface bram port=beta storage_type=ram_t2p
    #pragma HLS interface bram port=output_data storage_type=ram_t2p

    #pragma HLS dataflow

    GroupNorm3D<F_MAP_0, INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH>(
        input_data, gamma, beta, output_data
    );
}