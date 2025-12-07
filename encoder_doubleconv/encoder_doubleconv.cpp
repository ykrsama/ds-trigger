#include "encoder_doubleconv.h"

void EncoderDoubleConv(
    data_t input[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    data_t kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    data_t gamma1[F_MAP_0],
    data_t beta1[F_MAP_0],
    data_t kernel2[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    data_t gamma2[F_MAP_0],
    data_t beta2[F_MAP_0],
    data_t output[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return

    #pragma HLS interface bram port=input storage_type=ram_t2p
    #pragma HLS interface bram port=kernel1 storage_type=ram_t2p
    #pragma HLS interface bram port=gamma1 storage_type=ram_t2p
    #pragma HLS interface bram port=beta1 storage_type=ram_t2p
    #pragma HLS interface bram port=kernel2 storage_type=ram_t2p
    #pragma HLS interface bram port=gamma2 storage_type=ram_t2p
    #pragma HLS interface bram port=beta2 storage_type=ram_t2p
    #pragma HLS interface bram port=output storage_type=ram_t2p

    #pragma HLS dataflow

    DoubleConv3D<F_MAP_0, F_MAP_0, F_MAP_1, POOL_OUTPUT_DEPTH, POOL_OUTPUT_HEIGHT, POOL_OUTPUT_WIDTH>(
        input, kernel1, gamma1, beta1, kernel2, gamma2, beta2, output
    );
}