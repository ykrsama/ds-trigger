#include "input_doubleconv.h"

// Top-level function that only calls DoubleConv3D2Head
void InputDoubleConv(
    // Input
    float input[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],

    // Weights for DoubleConv3D2Head
    float kernel1[F_MAP_h][IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float gamma1[IN_CHANNELS],
    float beta1[IN_CHANNELS],
    float kernel2[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float gamma2[F_MAP_h],
    float beta2[F_MAP_h],

    // Outputs
    float output1[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output2[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return

    #pragma HLS interface bram port=input
    #pragma HLS bind_storage variable=input type=ram_t2p impl=bram

    #pragma HLS interface bram port=kernel1
    #pragma HLS bind_storage variable=kernel1 type=ram_t2p impl=bram

    #pragma HLS interface bram port=gamma1
    #pragma HLS bind_storage variable=gamma1 type=ram_t2p impl=bram

    #pragma HLS interface bram port=beta1
    #pragma HLS bind_storage variable=beta1 type=ram_t2p impl=bram

    #pragma HLS interface bram port=kernel2
    #pragma HLS bind_storage variable=kernel2 type=ram_t2p impl=bram

    #pragma HLS interface bram port=gamma2
    #pragma HLS bind_storage variable=gamma2 type=ram_t2p impl=bram

    #pragma HLS interface bram port=beta2
    #pragma HLS bind_storage variable=beta2 type=ram_t2p impl=bram

    #pragma HLS interface bram port=output1
    #pragma HLS bind_storage variable=output1 type=ram_t2p impl=bram

    #pragma HLS interface bram port=output2
    #pragma HLS bind_storage variable=output2 type=ram_t2p impl=bram

    #pragma HLS dataflow

    DoubleConv3D2Head<IN_CHANNELS, F_MAP_h, F_MAP_0>(
        input,
        kernel1, gamma1, beta1,
        kernel2, gamma2, beta2,
        output1,
        output2
    );

}