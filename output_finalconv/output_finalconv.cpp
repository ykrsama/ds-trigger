#include "output_finalconv.h"

void OutputFinalConv(
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float kernel[OUT_CHANNELS][F_MAP_0][1][1][1],
    float bias[OUT_CHANNELS],
    float output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return

    #pragma HLS interface bram port=input
    #pragma HLS bind_storage variable=input type=ram_t2p impl=bram

    #pragma HLS interface bram port=kernel
    #pragma HLS bind_storage variable=kernel type=ram_t2p impl=bram

    #pragma HLS interface bram port=bias
    #pragma HLS bind_storage variable=bias type=ram_t2p impl=bram

    #pragma HLS interface bram port=output
    #pragma HLS bind_storage variable=output type=ram_t2p impl=bram

    #pragma HLS dataflow

    FinalConv1x1<F_MAP_0, OUT_CHANNELS>(
        kernel, bias, input, output
    );
}