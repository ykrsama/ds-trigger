#include "doubleconv_component.h"
#include "../template/unet3d_reduced.h"

// Template function instantiations for DoubleConv3D
template void GroupNorm3D<IN_CHANNELS>(float[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[IN_CHANNELS], float[IN_CHANNELS], float[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
template void GroupNorm3D<F_MAP_h>(float[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[F_MAP_h], float[F_MAP_h], float[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

template void Conv3D<IN_CHANNELS, F_MAP_h>(float[F_MAP_h][IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
template void Conv3D<F_MAP_h, F_MAP_0>(float[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

// Top-level function that only calls DoubleConv3D2Head
void DoubleConvComponent(
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

    GroupNorm3D<IN_CHANNELS>(input, gamma1, beta1, gn1_out);
    Conv3D<IN_CHANNELS, F_MAP_h>(kernel1, gn1_out, conv1_out);
    GroupNorm3D<F_MAP_h>(conv1_out, gamma2, beta2, gn2_out);
    Conv3D<F_MAP_h, F_MAP_0>(kernel2, gn2_out, temp_output);

    // Duplicate output to both streams for skip connection
    // TODO: Review here
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int ch = 0; ch < T_OUT_CHANNELS; ch++) {
            for (int depth = 0; depth < T_INPUT_DEPTH; depth++) {
                for (int height = 0; height < T_INPUT_HEIGHT; height++) {
                    #pragma HLS pipeline II=1
                    for (int width = 0; width < T_INPUT_WIDTH; width++) {
                        float value = temp_output[batch][ch][depth][height][width];
                        output1[batch][ch][depth][height][width] = value;
                        output2[batch][ch][depth][height][width] = value;
                    }
                }
            }
        }
    }
}