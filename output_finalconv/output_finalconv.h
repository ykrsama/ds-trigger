#ifndef OUTPUT_FINALCONV_H
#define OUTPUT_FINALCONV_H

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include "../template/model.h"

void FinalConv1x1(
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float kernel[OUT_CHANNELS][F_MAP_0][1][1][1],
    float bias[OUT_CHANNELS],
    float output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

#endif // OUTPUT_FINALCONV_H