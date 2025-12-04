#include "encoder_doubleconv.h"

void EncoderDoubleConv(
    float input[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    float kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float gamma1[F_MAP_0],
    float beta1[F_MAP_0],
    float kernel2[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float gamma2[F_MAP_0],
    float beta2[F_MAP_0],
    float output[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
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

    #pragma HLS interface bram port=output
    #pragma HLS bind_storage variable=output type=ram_t2p impl=bram

    #pragma HLS dataflow

    DoubleConv3D<F_MAP_0, F_MAP_0, F_MAP_1, POOL_OUTPUT_DEPTH, POOL_OUTPUT_HEIGHT, POOL_OUTPUT_WIDTH>(
        input, kernel1, gamma1, beta1, kernel2, gamma2, beta2, output
    );
}