#include "test_conv3d.h"

void TestConv3D(
    float kernel[F_MAP_0][IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return

    #pragma HLS interface bram port=kernel storage_type=ram_t2p
    #pragma HLS interface bram port=input storage_type=ram_t2p
    #pragma HLS interface bram port=output storage_type=ram_t2p

    #pragma HLS dataflow

    Conv3D<IN_CHANNELS, F_MAP_0, INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH>(
        kernel, input, output
    );
}