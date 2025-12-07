#include "test_conv.h"

void  TestConv(
    data_t kernel[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    data_t input[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    data_t output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return

    #pragma HLS interface bram port=kernel storage_type=ram_t2p
    #pragma HLS interface bram port=input storage_type=ram_t2p
    #pragma HLS interface bram port=output storage_type=ram_t2p

    #pragma HLS dataflow

    Conv3D<F_MAP_h, F_MAP_0>(
        kernel, input, output
    );
}