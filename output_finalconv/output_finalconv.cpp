#include "output_finalconv.h"

void OutputFinalConv(
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float kernel[OUT_CHANNELS][F_MAP_0][1][1][1],
    float bias[OUT_CHANNELS],
    float output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return

    #pragma HLS interface bram port=input storage_type=ram_t2p
    #pragma HLS interface bram port=kernel storage_type=ram_t2p
    #pragma HLS interface bram port=bias storage_type=ram_t2p
    #pragma HLS interface bram port=output storage_type=ram_t2p

    #pragma HLS dataflow

    FinalConv1x1<F_MAP_0, OUT_CHANNELS>(
        kernel, bias, input, output
    );
}