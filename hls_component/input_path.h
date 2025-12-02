#ifndef INPUT_PATH_H
#define INPUT_PATH_H

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include "model_parameter.h"

// Input path function declarations
void InputConv3D_1(
    float kernel[F_MAP_h][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void InputConv3D_2(
    float kernel[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

// Input path function implementations

void InputConv3D_1(
    float kernel[F_MAP_h][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS array_partition variable=kernel cyclic factor=INPUT_CHANNELS dim=2
    #pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=5

    int PADDED_DEPTH = INPUT_DEPTH + 2 * CONV_PADDING;
    int PADDED_HEIGHT = INPUT_HEIGHT + 2 * CONV_PADDING;
    int PADDED_WIDTH = INPUT_WIDTH + 2 * CONV_PADDING;

    float padded_input[BATCH_SIZE][INPUT_CHANNELS][PADDED_DEPTH][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS stream variable=padded_input depth=10 type=fifo
    #pragma HLS bind_storage variable=padded_input type=ram_t2p impl=bram

    // Padding operation
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {
                    for (int in_ch = 0; in_ch < INPUT_CHANNELS; in_ch++) {
                        float pad_value = 0.0f;
                        int orig_depth = depth - CONV_PADDING;
                        int orig_height = height - CONV_PADDING;
                        int orig_width = width - CONV_PADDING;

                        if (orig_depth >= 0 && orig_depth < INPUT_DEPTH &&
                            orig_height >= 0 && orig_height < INPUT_HEIGHT &&
                            orig_width >= 0 && orig_width < INPUT_WIDTH) {
                            pad_value = input[batch][in_ch][orig_depth][orig_height][orig_width];
                        }
                        padded_input[batch][in_ch][depth][height][width] = pad_value;
                    }
                }
            }
        }
    }

    // Buffer definitions for convolution
    float cube_buffer[BATCH_SIZE][INPUT_CHANNELS][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS bind_storage variable=cube_buffer type=ram_2p impl=lutram

    float line_buffer[BATCH_SIZE][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
    #pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

    float window_buffer[BATCH_SIZE][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    #pragma HLS array_partition variable=window_buffer cyclic factor=INPUT_CHANNELS dim=2
    #pragma HLS array_partition variable=window_buffer cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=window_buffer cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=window_buffer cyclic factor=CONV_KERNEL dim=5
    #pragma HLS bind_storage variable=window_buffer type=ram_2p impl=lutram

    // Convolution computation
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {

                    // Update cube buffer
                    for (int in_ch = 0; in_ch < INPUT_CHANNELS; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer[batch][in_ch][kd][height][width] =
                                cube_buffer[batch][in_ch][kd + 1][height][width];
                        }
                        cube_buffer[batch][in_ch][CONV_KERNEL - 1][height][width] =
                            padded_input[batch][in_ch][depth][height][width];
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < INPUT_CHANNELS; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL - 1; kh++) {
                                    line_buffer[batch][in_ch][kd][kh][width] =
                                        line_buffer[batch][in_ch][kd][kh + 1][width];
                                }
                                line_buffer[batch][in_ch][kd][CONV_KERNEL - 1][width] =
                                    cube_buffer[batch][in_ch][kd][height][width];
                            }
                        }

                        if (height >= CONV_KERNEL - 1) {
                            // Update window buffer
                            for (int in_ch = 0; in_ch < INPUT_CHANNELS; in_ch++) {
                                for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                    for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                        for (int kw = 0; kw < CONV_KERNEL - 1; kw++) {
                                            window_buffer[batch][in_ch][kd][kh][kw] =
                                                window_buffer[batch][in_ch][kd][kh][kw + 1];
                                        }
                                        window_buffer[batch][in_ch][kd][kh][CONV_KERNEL - 1] =
                                            line_buffer[batch][in_ch][kd][kh][width];
                                    }
                                }
                            }

                            // Convolution computation
                            if (width >= CONV_KERNEL - 1 &&
                                (depth - CONV_KERNEL + 1) % 1 == 0 &&
                                (height - CONV_KERNEL + 1) % 1 == 0 &&
                                (width - CONV_KERNEL + 1) % 1 == 0) {

                                float accum[F_MAP_h];
                                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < F_MAP_h; out_ch++) {
                                    #pragma HLS pipeline II=1
                                    accum[out_ch] = 0.0f;

                                    for (int in_ch = 0; in_ch < INPUT_CHANNELS; in_ch++) {
                                        for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                                    float window_val = window_buffer[batch][in_ch][kd][kh][kw];
                                                    float kernel_val = kernel[out_ch][in_ch][kd][kh][kw];
                                                    accum[out_ch] += window_val * kernel_val;
                                                }
                                            }
                                        }
                                    }

                                    int out_depth = (depth - CONV_KERNEL) / 1 + 1;
                                    int out_height = (height - CONV_KERNEL) / 1 + 1;
                                    int out_width = (width - CONV_KERNEL) / 1 + 1;

                                    // ReLU activation
                                    float output_value = accum[out_ch];
                                    bool is_positive = output_value > 0.0f;
                                    float relu_output = is_positive ? output_value : 0.0f;
                                    output[batch][out_ch][out_depth][out_height][out_width] = relu_output;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif // INPUT_PATH_H