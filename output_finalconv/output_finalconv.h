#ifndef OUTPUT_FINALCONV_H
#define OUTPUT_FINALCONV_H

#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include "../template/model.h"

void OutputFinalConv(
    float input[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float kernel[OUT_CHANNELS][IN_CHANNELS][1][1][1],
    float bias[OUT_CHANNELS],
    float output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

#endif // OUTPUT_FINALCONV_H