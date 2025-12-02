#include "unet3d_reduced.h"

// Input convolution block: INPUT_CHANNELS -> F_MAP_0 (1 -> 64)
// Implements two 3x3 convolutions with ReLU activation
void InputConv3D(
    float kernel1[F_MAP_h][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float kernel2[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS array_partition variable=kernel1 cyclic factor=INPUT_CHANNELS dim=2
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONV_KERNEL dim=5

    #pragma HLS array_partition variable=kernel2 cyclic factor=F_MAP_h dim=2
    #pragma HLS array_partition variable=kernel2 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=kernel2 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=kernel2 cyclic factor=CONV_KERNEL dim=5

    // Padded dimensions
    int PADDED_DEPTH = INPUT_DEPTH + 2 * CONV_PADDING;
    int PADDED_HEIGHT = INPUT_HEIGHT + 2 * CONV_PADDING;
    int PADDED_WIDTH = INPUT_WIDTH + 2 * CONV_PADDING;

    // Padded input
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

    // First convolution output
    float conv1_out[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=bram

    // Buffer definitions for first convolution
    float cube_buffer1[BATCH_SIZE][INPUT_CHANNELS][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS bind_storage variable=cube_buffer1 type=ram_2p impl=lutram

    float line_buffer1[BATCH_SIZE][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
    #pragma HLS bind_storage variable=line_buffer1 type=ram_2p impl=lutram

    float window_buffer1[BATCH_SIZE][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=INPUT_CHANNELS dim=2
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONV_KERNEL dim=5
    #pragma HLS bind_storage variable=window_buffer1 type=ram_2p impl=lutram

    // First convolution: INPUT_CHANNELS -> F_MAP_h
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {

                    // Update cube buffer
                    for (int in_ch = 0; in_ch < INPUT_CHANNELS; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer1[batch][in_ch][kd][height][width] =
                                cube_buffer1[batch][in_ch][kd + 1][height][width];
                        }
                        cube_buffer1[batch][in_ch][CONV_KERNEL - 1][height][width] =
                            padded_input[batch][in_ch][depth][height][width];
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < INPUT_CHANNELS; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL - 1; kh++) {
                                    line_buffer1[batch][in_ch][kd][kh][width] =
                                        line_buffer1[batch][in_ch][kd][kh + 1][width];
                                }
                                line_buffer1[batch][in_ch][kd][CONV_KERNEL - 1][width] =
                                    cube_buffer1[batch][in_ch][kd][height][width];
                            }
                        }

                        if (height >= CONV_KERNEL - 1) {
                            // Update window buffer
                            for (int in_ch = 0; in_ch < INPUT_CHANNELS; in_ch++) {
                                for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                    for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                        for (int kw = 0; kw < CONV_KERNEL - 1; kw++) {
                                            window_buffer1[batch][in_ch][kd][kh][kw] =
                                                window_buffer1[batch][in_ch][kd][kh][kw + 1];
                                        }
                                        window_buffer1[batch][in_ch][kd][kh][CONV_KERNEL - 1] =
                                            line_buffer1[batch][in_ch][kd][kh][width];
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
                                                    float window_val = window_buffer1[batch][in_ch][kd][kh][kw];
                                                    float kernel_val = kernel1[out_ch][in_ch][kd][kh][kw];
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
                                    conv1_out[batch][out_ch][out_depth][out_height][out_width] = relu_output;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Second convolution: F_MAP_h -> F_MAP_0 (similar structure but different dimensions)
    // Buffer definitions for second convolution
    float cube_buffer2[BATCH_SIZE][F_MAP_h][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS bind_storage variable=cube_buffer2 type=ram_2p impl=lutram

    float line_buffer2[BATCH_SIZE][F_MAP_h][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
    #pragma HLS bind_storage variable=line_buffer2 type=ram_2p impl=lutram

    float window_buffer2[BATCH_SIZE][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=F_MAP_h dim=2
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=CONV_KERNEL dim=5
    #pragma HLS bind_storage variable=window_buffer2 type=ram_2p impl=lutram

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {

                    // Update cube buffer
                    for (int in_ch = 0; in_ch < F_MAP_h; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer2[batch][in_ch][kd][height][width] =
                                cube_buffer2[batch][in_ch][kd + 1][height][width];
                        }
                        // Use padded conv1_out (need to add padding)
                        float pad_value = 0.0f;
                        int orig_depth = depth - CONV_PADDING;
                        int orig_height = height - CONV_PADDING;
                        int orig_width = width - CONV_PADDING;

                        if (orig_depth >= 0 && orig_depth < INPUT_DEPTH &&
                            orig_height >= 0 && orig_height < INPUT_HEIGHT &&
                            orig_width >= 0 && orig_width < INPUT_WIDTH) {
                            pad_value = conv1_out[batch][in_ch][orig_depth][orig_height][orig_width];
                        }
                        cube_buffer2[batch][in_ch][CONV_KERNEL - 1][height][width] = pad_value;
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < F_MAP_h; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL - 1; kh++) {
                                    line_buffer2[batch][in_ch][kd][kh][width] =
                                        line_buffer2[batch][in_ch][kd][kh + 1][width];
                                }
                                line_buffer2[batch][in_ch][kd][CONV_KERNEL - 1][width] =
                                    cube_buffer2[batch][in_ch][kd][height][width];
                            }
                        }

                        if (height >= CONV_KERNEL - 1) {
                            // Update window buffer
                            for (int in_ch = 0; in_ch < F_MAP_h; in_ch++) {
                                for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                    for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                        for (int kw = 0; kw < CONV_KERNEL - 1; kw++) {
                                            window_buffer2[batch][in_ch][kd][kh][kw] =
                                                window_buffer2[batch][in_ch][kd][kh][kw + 1];
                                        }
                                        window_buffer2[batch][in_ch][kd][kh][CONV_KERNEL - 1] =
                                            line_buffer2[batch][in_ch][kd][kh][width];
                                    }
                                }
                            }

                            // Convolution computation
                            if (width >= CONV_KERNEL - 1 &&
                                (depth - CONV_KERNEL + 1) % 1 == 0 &&
                                (height - CONV_KERNEL + 1) % 1 == 0 &&
                                (width - CONV_KERNEL + 1) % 1 == 0) {

                                float accum[F_MAP_0];
                                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
                                    #pragma HLS pipeline II=1
                                    accum[out_ch] = 0.0f;

                                    for (int in_ch = 0; in_ch < F_MAP_h; in_ch++) {
                                        for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                                    float window_val = window_buffer2[batch][in_ch][kd][kh][kw];
                                                    float kernel_val = kernel2[out_ch][in_ch][kd][kh][kw];
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

// MaxPool3D block: spatial downsampling by 2
void EncoderMaxPool3D(
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
) {
    // Buffer definitions
    float cube_buffer[BATCH_SIZE][F_MAP_0][POOL_KERNEL][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=cube_buffer type=ram_2p impl=lutram

    float line_buffer[BATCH_SIZE][F_MAP_0][POOL_KERNEL][POOL_KERNEL][INPUT_WIDTH];
    #pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

    float window_buffer[BATCH_SIZE][F_MAP_0][POOL_KERNEL][POOL_KERNEL][POOL_KERNEL];
    #pragma HLS array_partition variable=window_buffer cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=window_buffer cyclic factor=POOL_KERNEL dim=3
    #pragma HLS array_partition variable=window_buffer cyclic factor=POOL_KERNEL dim=4
    #pragma HLS array_partition variable=window_buffer cyclic factor=POOL_KERNEL dim=5
    #pragma HLS bind_storage variable=window_buffer type=ram_2p impl=lutram

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {

                    // Update cube buffer
                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                        for (int kd = 0; kd < POOL_KERNEL - 1; kd++) {
                            cube_buffer[batch][in_ch][kd][height][width] =
                                cube_buffer[batch][in_ch][kd + 1][height][width];
                        }
                        cube_buffer[batch][in_ch][POOL_KERNEL - 1][height][width] =
                            input[batch][in_ch][depth][height][width];
                    }

                    if (depth >= POOL_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            for (int kd = 0; kd < POOL_KERNEL; kd++) {
                                for (int kh = 0; kh < POOL_KERNEL - 1; kh++) {
                                    line_buffer[batch][in_ch][kd][kh][width] =
                                        line_buffer[batch][in_ch][kd][kh + 1][width];
                                }
                                line_buffer[batch][in_ch][kd][POOL_KERNEL - 1][width] =
                                    cube_buffer[batch][in_ch][kd][height][width];
                            }
                        }

                        if (height >= POOL_KERNEL - 1) {
                            // Update window buffer
                            for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                                for (int kd = 0; kd < POOL_KERNEL; kd++) {
                                    for (int kh = 0; kh < POOL_KERNEL; kh++) {
                                        for (int kw = 0; kw < POOL_KERNEL - 1; kw++) {
                                            window_buffer[batch][in_ch][kd][kh][kw] =
                                                window_buffer[batch][in_ch][kd][kh][kw + 1];
                                        }
                                        window_buffer[batch][in_ch][kd][kh][POOL_KERNEL - 1] =
                                            line_buffer[batch][in_ch][kd][kh][width];
                                    }
                                }
                            }

                            // Pooling computation
                            if (width >= POOL_KERNEL - 1 &&
                                (depth - POOL_KERNEL + 1) % POOL_STRIDE == 0 &&
                                (height - POOL_KERNEL + 1) % POOL_STRIDE == 0 &&
                                (width - POOL_KERNEL + 1) % POOL_STRIDE == 0) {

                                float max_vals[F_MAP_0];
                                #pragma HLS bind_storage variable=max_vals type=ram_2p impl=bram

                                for (int ch = 0; ch < F_MAP_0; ch++) {
                                    #pragma HLS pipeline II=1
                                    max_vals[ch] = -1e9f;

                                    for (int kd = 0; kd < POOL_KERNEL; kd++) {
                                        for (int kh = 0; kh < POOL_KERNEL; kh++) {
                                            for (int kw = 0; kw < POOL_KERNEL; kw++) {
                                                float window_val = window_buffer[batch][ch][kd][kh][kw];
                                                if (window_val > max_vals[ch]) {
                                                    max_vals[ch] = window_val;
                                                }
                                            }
                                        }
                                    }

                                    int out_depth = (depth - POOL_KERNEL + 1) / POOL_STRIDE;
                                    int out_height = (height - POOL_KERNEL + 1) / POOL_STRIDE;
                                    int out_width = (width - POOL_KERNEL + 1) / POOL_STRIDE;

                                    output[batch][ch][out_depth][out_height][out_width] = max_vals[ch];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Encoder convolution block: F_MAP_0 -> F_MAP_1 (64 -> 128)
void EncoderConv3D(
    float kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float kernel2[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
) {
    #pragma HLS array_partition variable=kernel1 cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONV_KERNEL dim=5

    #pragma HLS array_partition variable=kernel2 cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=kernel2 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=kernel2 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=kernel2 cyclic factor=CONV_KERNEL dim=5

    // Padded dimensions
    int PADDED_DEPTH = POOL_OUTPUT_DEPTH + 2 * CONV_PADDING;
    int PADDED_HEIGHT = POOL_OUTPUT_HEIGHT + 2 * CONV_PADDING;
    int PADDED_WIDTH = POOL_OUTPUT_WIDTH + 2 * CONV_PADDING;

    // Padded input
    float padded_input[BATCH_SIZE][F_MAP_0][PADDED_DEPTH][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS stream variable=padded_input depth=10 type=fifo
    #pragma HLS bind_storage variable=padded_input type=ram_t2p impl=bram

    // Padding operation
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {
                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                        float pad_value = 0.0f;
                        int orig_depth = depth - CONV_PADDING;
                        int orig_height = height - CONV_PADDING;
                        int orig_width = width - CONV_PADDING;

                        if (orig_depth >= 0 && orig_depth < POOL_OUTPUT_DEPTH &&
                            orig_height >= 0 && orig_height < POOL_OUTPUT_HEIGHT &&
                            orig_width >= 0 && orig_width < POOL_OUTPUT_WIDTH) {
                            pad_value = input[batch][in_ch][orig_depth][orig_height][orig_width];
                        }
                        padded_input[batch][in_ch][depth][height][width] = pad_value;
                    }
                }
            }
        }
    }

    // First convolution output
    float conv1_out[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=bram

    // Buffer definitions for first convolution
    float cube_buffer1[BATCH_SIZE][F_MAP_0][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS bind_storage variable=cube_buffer1 type=ram_2p impl=lutram

    float line_buffer1[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
    #pragma HLS bind_storage variable=line_buffer1 type=ram_2p impl=lutram

    float window_buffer1[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONV_KERNEL dim=5
    #pragma HLS bind_storage variable=window_buffer1 type=ram_2p impl=lutram

    // First convolution: F_MAP_0 -> F_MAP_0
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {

                    // Update cube buffer
                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer1[batch][in_ch][kd][height][width] =
                                cube_buffer1[batch][in_ch][kd + 1][height][width];
                        }
                        cube_buffer1[batch][in_ch][CONV_KERNEL - 1][height][width] =
                            padded_input[batch][in_ch][depth][height][width];
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL - 1; kh++) {
                                    line_buffer1[batch][in_ch][kd][kh][width] =
                                        line_buffer1[batch][in_ch][kd][kh + 1][width];
                                }
                                line_buffer1[batch][in_ch][kd][CONV_KERNEL - 1][width] =
                                    cube_buffer1[batch][in_ch][kd][height][width];
                            }
                        }

                        if (height >= CONV_KERNEL - 1) {
                            // Update window buffer
                            for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                                for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                    for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                        for (int kw = 0; kw < CONV_KERNEL - 1; kw++) {
                                            window_buffer1[batch][in_ch][kd][kh][kw] =
                                                window_buffer1[batch][in_ch][kd][kh][kw + 1];
                                        }
                                        window_buffer1[batch][in_ch][kd][kh][CONV_KERNEL - 1] =
                                            line_buffer1[batch][in_ch][kd][kh][width];
                                    }
                                }
                            }

                            // Convolution computation
                            if (width >= CONV_KERNEL - 1 &&
                                (depth - CONV_KERNEL + 1) % 1 == 0 &&
                                (height - CONV_KERNEL + 1) % 1 == 0 &&
                                (width - CONV_KERNEL + 1) % 1 == 0) {

                                float accum[F_MAP_0];
                                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
                                    #pragma HLS pipeline II=1
                                    accum[out_ch] = 0.0f;

                                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                                        for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                                    float window_val = window_buffer1[batch][in_ch][kd][kh][kw];
                                                    float kernel_val = kernel1[out_ch][in_ch][kd][kh][kw];
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
                                    conv1_out[batch][out_ch][out_depth][out_height][out_width] = relu_output;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Second convolution: F_MAP_0 -> F_MAP_1
    // Buffer definitions for second convolution
    float cube_buffer2[BATCH_SIZE][F_MAP_0][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS bind_storage variable=cube_buffer2 type=ram_2p impl=lutram

    float line_buffer2[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
    #pragma HLS bind_storage variable=line_buffer2 type=ram_2p impl=lutram

    float window_buffer2[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=CONV_KERNEL dim=5
    #pragma HLS bind_storage variable=window_buffer2 type=ram_2p impl=lutram

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {

                    // Update cube buffer
                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer2[batch][in_ch][kd][height][width] =
                                cube_buffer2[batch][in_ch][kd + 1][height][width];
                        }
                        // Use padded conv1_out
                        float pad_value = 0.0f;
                        int orig_depth = depth - CONV_PADDING;
                        int orig_height = height - CONV_PADDING;
                        int orig_width = width - CONV_PADDING;

                        if (orig_depth >= 0 && orig_depth < POOL_OUTPUT_DEPTH &&
                            orig_height >= 0 && orig_height < POOL_OUTPUT_HEIGHT &&
                            orig_width >= 0 && orig_width < POOL_OUTPUT_WIDTH) {
                            pad_value = conv1_out[batch][in_ch][orig_depth][orig_height][orig_width];
                        }
                        cube_buffer2[batch][in_ch][CONV_KERNEL - 1][height][width] = pad_value;
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL - 1; kh++) {
                                    line_buffer2[batch][in_ch][kd][kh][width] =
                                        line_buffer2[batch][in_ch][kd][kh + 1][width];
                                }
                                line_buffer2[batch][in_ch][kd][CONV_KERNEL - 1][width] =
                                    cube_buffer2[batch][in_ch][kd][height][width];
                            }
                        }

                        if (height >= CONV_KERNEL - 1) {
                            // Update window buffer
                            for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                                for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                    for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                        for (int kw = 0; kw < CONV_KERNEL - 1; kw++) {
                                            window_buffer2[batch][in_ch][kd][kh][kw] =
                                                window_buffer2[batch][in_ch][kd][kh][kw + 1];
                                        }
                                        window_buffer2[batch][in_ch][kd][kh][CONV_KERNEL - 1] =
                                            line_buffer2[batch][in_ch][kd][kh][width];
                                    }
                                }
                            }

                            // Convolution computation
                            if (width >= CONV_KERNEL - 1 &&
                                (depth - CONV_KERNEL + 1) % 1 == 0 &&
                                (height - CONV_KERNEL + 1) % 1 == 0 &&
                                (width - CONV_KERNEL + 1) % 1 == 0) {

                                float accum[F_MAP_1];
                                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < F_MAP_1; out_ch++) {
                                    #pragma HLS pipeline II=1
                                    accum[out_ch] = 0.0f;

                                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                                        for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                                    float window_val = window_buffer2[batch][in_ch][kd][kh][kw];
                                                    float kernel_val = kernel2[out_ch][in_ch][kd][kh][kw];
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

// Nearest neighbor upsampling: spatial upsampling by 2
void DecoderUpsample3D(
    float input[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS array_partition variable=input cyclic factor=F_MAP_1 dim=2
    #pragma HLS array_partition variable=output cyclic factor=F_MAP_1 dim=2
    #pragma HLS bind_storage variable=input type=ram_t2p impl=bram
    #pragma HLS bind_storage variable=output type=ram_t2p impl=bram

    // Line buffer for streaming data access pattern
    float line_buffer[F_MAP_1][POOL_OUTPUT_WIDTH];
    #pragma HLS array_partition variable=line_buffer cyclic factor=F_MAP_1 dim=1
    #pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

    // Main processing loops
    batch_loop: for (int batch = 0; batch < BATCH_SIZE; batch++) {
        #pragma HLS loop_tripcount min=1 max=1

        depth_loop: for (int src_d = 0; src_d < POOL_OUTPUT_DEPTH; src_d++) {
            #pragma HLS pipeline off
            #pragma HLS loop_tripcount min=16 max=16

            height_loop: for (int src_h = 0; src_h < POOL_OUTPUT_HEIGHT; src_h++) {
                #pragma HLS pipeline off

                // Load input line into buffer for reuse
                load_loop: for (int src_w = 0; src_w < POOL_OUTPUT_WIDTH; src_w++) {
                    #pragma HLS pipeline II=1
                    #pragma HLS unroll factor=2

                    for (int ch = 0; ch < F_MAP_1; ch++) {
                        #pragma HLS unroll
                        line_buffer[ch][src_w] = input[batch][ch][src_d][src_h][src_w];
                    }
                }

                // Generate upsampled outputs (2x2 expansion per source pixel)
                upsample_height_loop: for (int h_offset = 0; h_offset < 2; h_offset++) {
                    #pragma HLS pipeline off

                    int out_h = src_h * 2 + h_offset;
                    if (out_h < INPUT_HEIGHT) {

                        upsample_width_loop: for (int src_w = 0; src_w < POOL_OUTPUT_WIDTH; src_w++) {
                            #pragma HLS pipeline II=1
                            #pragma HLS unroll factor=2

                            int out_w_base = src_w * 2;

                            // Process all channels in parallel
                            channel_loop: for (int ch = 0; ch < F_MAP_1; ch++) {
                                #pragma HLS unroll

                                float pixel_val = line_buffer[ch][src_w];

                                // Generate 2x upsampling in width dimension
                                if (out_w_base < INPUT_WIDTH) {
                                    // First depth output
                                    int out_d1 = src_d * 2;
                                    if (out_d1 < INPUT_DEPTH) {
                                        output[batch][ch][out_d1][out_h][out_w_base] = pixel_val;
                                    }

                                    // Second depth output
                                    int out_d2 = src_d * 2 + 1;
                                    if (out_d2 < INPUT_DEPTH) {
                                        output[batch][ch][out_d2][out_h][out_w_base] = pixel_val;
                                    }
                                }

                                if (out_w_base + 1 < INPUT_WIDTH) {
                                    // First depth output
                                    int out_d1 = src_d * 2;
                                    if (out_d1 < INPUT_DEPTH) {
                                        output[batch][ch][out_d1][out_h][out_w_base + 1] = pixel_val;
                                    }

                                    // Second depth output
                                    int out_d2 = src_d * 2 + 1;
                                    if (out_d2 < INPUT_DEPTH) {
                                        output[batch][ch][out_d2][out_h][out_w_base + 1] = pixel_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Concatenation function
void ConcatenateTensors(
    float tensor1[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float tensor2[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS array_partition variable=tensor1 cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=tensor2 cyclic factor=F_MAP_1 dim=2
    #pragma HLS array_partition variable=output cyclic factor=CONCAT_CHANNELS dim=2
    #pragma HLS bind_storage variable=tensor1 type=ram_t2p impl=bram
    #pragma HLS bind_storage variable=tensor2 type=ram_t2p impl=bram
    #pragma HLS bind_storage variable=output type=ram_t2p impl=bram

    batch_loop: for (int batch = 0; batch < BATCH_SIZE; batch++) {
        #pragma HLS loop_tripcount min=1 max=1

        depth_loop: for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            #pragma HLS pipeline off
            #pragma HLS loop_tripcount min=32 max=32

            height_loop: for (int height = 0; height < INPUT_HEIGHT; height++) {
                #pragma HLS pipeline off

                width_loop: for (int width = 0; width < INPUT_WIDTH; width++) {
                    #pragma HLS pipeline II=1
                    #pragma HLS unroll factor=4

                    // Process tensor1 channels in parallel (F_MAP_0 = 64 channels)
                    tensor1_channels: for (int ch = 0; ch < F_MAP_0; ch++) {
                        #pragma HLS unroll
                        output[batch][ch][depth][height][width] =
                            tensor1[batch][ch][depth][height][width];
                    }

                    // Process tensor2 channels in parallel (F_MAP_1 = 128 channels)
                    tensor2_channels: for (int ch = 0; ch < F_MAP_1; ch++) {
                        #pragma HLS unroll
                        output[batch][F_MAP_0 + ch][depth][height][width] =
                            tensor2[batch][ch][depth][height][width];
                    }
                }
            }
        }
    }
}

// Decoder convolution block: CONCAT_CHANNELS -> F_MAP_0 (192 -> 64)
void DecoderConv3D(
    float kernel1[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float kernel2[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONCAT_CHANNELS dim=2
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONV_KERNEL dim=5

    #pragma HLS array_partition variable=kernel2 cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=kernel2 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=kernel2 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=kernel2 cyclic factor=CONV_KERNEL dim=5

    // Padded dimensions
    int PADDED_DEPTH = INPUT_DEPTH + 2 * CONV_PADDING;
    int PADDED_HEIGHT = INPUT_HEIGHT + 2 * CONV_PADDING;
    int PADDED_WIDTH = INPUT_WIDTH + 2 * CONV_PADDING;

    // Padded input
    float padded_input[BATCH_SIZE][CONCAT_CHANNELS][PADDED_DEPTH][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS stream variable=padded_input depth=10 type=fifo
    #pragma HLS bind_storage variable=padded_input type=ram_t2p impl=bram

    // Padding operation
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {
                    for (int in_ch = 0; in_ch < CONCAT_CHANNELS; in_ch++) {
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

    // First convolution output
    float conv1_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=bram

    // Buffer definitions for first convolution
    float cube_buffer1[BATCH_SIZE][CONCAT_CHANNELS][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS bind_storage variable=cube_buffer1 type=ram_2p impl=lutram

    float line_buffer1[BATCH_SIZE][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
    #pragma HLS bind_storage variable=line_buffer1 type=ram_2p impl=lutram

    float window_buffer1[BATCH_SIZE][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONCAT_CHANNELS dim=2
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONV_KERNEL dim=5
    #pragma HLS bind_storage variable=window_buffer1 type=ram_2p impl=lutram

    // First convolution: CONCAT_CHANNELS -> F_MAP_0
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {

                    // Update cube buffer
                    for (int in_ch = 0; in_ch < CONCAT_CHANNELS; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer1[batch][in_ch][kd][height][width] =
                                cube_buffer1[batch][in_ch][kd + 1][height][width];
                        }
                        cube_buffer1[batch][in_ch][CONV_KERNEL - 1][height][width] =
                            padded_input[batch][in_ch][depth][height][width];
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < CONCAT_CHANNELS; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL - 1; kh++) {
                                    line_buffer1[batch][in_ch][kd][kh][width] =
                                        line_buffer1[batch][in_ch][kd][kh + 1][width];
                                }
                                line_buffer1[batch][in_ch][kd][CONV_KERNEL - 1][width] =
                                    cube_buffer1[batch][in_ch][kd][height][width];
                            }
                        }

                        if (height >= CONV_KERNEL - 1) {
                            // Update window buffer
                            for (int in_ch = 0; in_ch < CONCAT_CHANNELS; in_ch++) {
                                for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                    for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                        for (int kw = 0; kw < CONV_KERNEL - 1; kw++) {
                                            window_buffer1[batch][in_ch][kd][kh][kw] =
                                                window_buffer1[batch][in_ch][kd][kh][kw + 1];
                                        }
                                        window_buffer1[batch][in_ch][kd][kh][CONV_KERNEL - 1] =
                                            line_buffer1[batch][in_ch][kd][kh][width];
                                    }
                                }
                            }

                            // Convolution computation
                            if (width >= CONV_KERNEL - 1 &&
                                (depth - CONV_KERNEL + 1) % 1 == 0 &&
                                (height - CONV_KERNEL + 1) % 1 == 0 &&
                                (width - CONV_KERNEL + 1) % 1 == 0) {

                                float accum[F_MAP_0];
                                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
                                    #pragma HLS pipeline II=1
                                    accum[out_ch] = 0.0f;

                                    for (int in_ch = 0; in_ch < CONCAT_CHANNELS; in_ch++) {
                                        for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                                    float window_val = window_buffer1[batch][in_ch][kd][kh][kw];
                                                    float kernel_val = kernel1[out_ch][in_ch][kd][kh][kw];
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
                                    conv1_out[batch][out_ch][out_depth][out_height][out_width] = relu_output;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Second convolution: F_MAP_0 -> F_MAP_0
    // Buffer definitions for second convolution
    float cube_buffer2[BATCH_SIZE][F_MAP_0][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS bind_storage variable=cube_buffer2 type=ram_2p impl=lutram

    float line_buffer2[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
    #pragma HLS bind_storage variable=line_buffer2 type=ram_2p impl=lutram

    float window_buffer2[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=CONV_KERNEL dim=5
    #pragma HLS bind_storage variable=window_buffer2 type=ram_2p impl=lutram

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {

                    // Update cube buffer
                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer2[batch][in_ch][kd][height][width] =
                                cube_buffer2[batch][in_ch][kd + 1][height][width];
                        }
                        // Use padded conv1_out
                        float pad_value = 0.0f;
                        int orig_depth = depth - CONV_PADDING;
                        int orig_height = height - CONV_PADDING;
                        int orig_width = width - CONV_PADDING;

                        if (orig_depth >= 0 && orig_depth < INPUT_DEPTH &&
                            orig_height >= 0 && orig_height < INPUT_HEIGHT &&
                            orig_width >= 0 && orig_width < INPUT_WIDTH) {
                            pad_value = conv1_out[batch][in_ch][orig_depth][orig_height][orig_width];
                        }
                        cube_buffer2[batch][in_ch][CONV_KERNEL - 1][height][width] = pad_value;
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL - 1; kh++) {
                                    line_buffer2[batch][in_ch][kd][kh][width] =
                                        line_buffer2[batch][in_ch][kd][kh + 1][width];
                                }
                                line_buffer2[batch][in_ch][kd][CONV_KERNEL - 1][width] =
                                    cube_buffer2[batch][in_ch][kd][height][width];
                            }
                        }

                        if (height >= CONV_KERNEL - 1) {
                            // Update window buffer
                            for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                                for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                    for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                        for (int kw = 0; kw < CONV_KERNEL - 1; kw++) {
                                            window_buffer2[batch][in_ch][kd][kh][kw] =
                                                window_buffer2[batch][in_ch][kd][kh][kw + 1];
                                        }
                                        window_buffer2[batch][in_ch][kd][kh][CONV_KERNEL - 1] =
                                            line_buffer2[batch][in_ch][kd][kh][width];
                                    }
                                }
                            }

                            // Convolution computation
                            if (width >= CONV_KERNEL - 1 &&
                                (depth - CONV_KERNEL + 1) % 1 == 0 &&
                                (height - CONV_KERNEL + 1) % 1 == 0 &&
                                (width - CONV_KERNEL + 1) % 1 == 0) {

                                float accum[F_MAP_0];
                                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
                                    #pragma HLS pipeline II=1
                                    accum[out_ch] = 0.0f;

                                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                                        for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                                    float window_val = window_buffer2[batch][in_ch][kd][kh][kw];
                                                    float kernel_val = kernel2[out_ch][in_ch][kd][kh][kw];
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

// Output convolution block: F_MAP_0 -> F_MAP_0 (64 -> 64)
void OutputConv3D(
    float kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float kernel2[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS array_partition variable=kernel1 cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=kernel1 cyclic factor=CONV_KERNEL dim=5

    #pragma HLS array_partition variable=kernel2 cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=kernel2 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=kernel2 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=kernel2 cyclic factor=CONV_KERNEL dim=5

    // Padded dimensions
    int PADDED_DEPTH = INPUT_DEPTH + 2 * CONV_PADDING;
    int PADDED_HEIGHT = INPUT_HEIGHT + 2 * CONV_PADDING;
    int PADDED_WIDTH = INPUT_WIDTH + 2 * CONV_PADDING;

    // Padded input
    float padded_input[BATCH_SIZE][F_MAP_0][PADDED_DEPTH][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS stream variable=padded_input depth=10 type=fifo
    #pragma HLS bind_storage variable=padded_input type=ram_t2p impl=bram

    // Padding operation
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {
                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
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

    // First convolution output
    float conv1_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=bram

    // Buffer definitions for first convolution
    float cube_buffer1[BATCH_SIZE][F_MAP_0][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS bind_storage variable=cube_buffer1 type=ram_2p impl=lutram

    float line_buffer1[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
    #pragma HLS bind_storage variable=line_buffer1 type=ram_2p impl=lutram

    float window_buffer1[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=window_buffer1 cyclic factor=CONV_KERNEL dim=5
    #pragma HLS bind_storage variable=window_buffer1 type=ram_2p impl=lutram

    // First convolution: F_MAP_0 -> F_MAP_0
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {

                    // Update cube buffer
                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer1[batch][in_ch][kd][height][width] =
                                cube_buffer1[batch][in_ch][kd + 1][height][width];
                        }
                        cube_buffer1[batch][in_ch][CONV_KERNEL - 1][height][width] =
                            padded_input[batch][in_ch][depth][height][width];
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL - 1; kh++) {
                                    line_buffer1[batch][in_ch][kd][kh][width] =
                                        line_buffer1[batch][in_ch][kd][kh + 1][width];
                                }
                                line_buffer1[batch][in_ch][kd][CONV_KERNEL - 1][width] =
                                    cube_buffer1[batch][in_ch][kd][height][width];
                            }
                        }

                        if (height >= CONV_KERNEL - 1) {
                            // Update window buffer
                            for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                                for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                    for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                        for (int kw = 0; kw < CONV_KERNEL - 1; kw++) {
                                            window_buffer1[batch][in_ch][kd][kh][kw] =
                                                window_buffer1[batch][in_ch][kd][kh][kw + 1];
                                        }
                                        window_buffer1[batch][in_ch][kd][kh][CONV_KERNEL - 1] =
                                            line_buffer1[batch][in_ch][kd][kh][width];
                                    }
                                }
                            }

                            // Convolution computation
                            if (width >= CONV_KERNEL - 1 &&
                                (depth - CONV_KERNEL + 1) % 1 == 0 &&
                                (height - CONV_KERNEL + 1) % 1 == 0 &&
                                (width - CONV_KERNEL + 1) % 1 == 0) {

                                float accum[F_MAP_0];
                                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
                                    #pragma HLS pipeline II=1
                                    accum[out_ch] = 0.0f;

                                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                                        for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                                    float window_val = window_buffer1[batch][in_ch][kd][kh][kw];
                                                    float kernel_val = kernel1[out_ch][in_ch][kd][kh][kw];
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
                                    conv1_out[batch][out_ch][out_depth][out_height][out_width] = relu_output;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Second convolution: F_MAP_0 -> F_MAP_0
    // Buffer definitions for second convolution
    float cube_buffer2[BATCH_SIZE][F_MAP_0][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS bind_storage variable=cube_buffer2 type=ram_2p impl=lutram

    float line_buffer2[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
    #pragma HLS bind_storage variable=line_buffer2 type=ram_2p impl=lutram

    float window_buffer2[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=CONV_KERNEL dim=5
    #pragma HLS bind_storage variable=window_buffer2 type=ram_2p impl=lutram

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {

                    // Update cube buffer
                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer2[batch][in_ch][kd][height][width] =
                                cube_buffer2[batch][in_ch][kd + 1][height][width];
                        }
                        // Use padded conv1_out
                        float pad_value = 0.0f;
                        int orig_depth = depth - CONV_PADDING;
                        int orig_height = height - CONV_PADDING;
                        int orig_width = width - CONV_PADDING;

                        if (orig_depth >= 0 && orig_depth < INPUT_DEPTH &&
                            orig_height >= 0 && orig_height < INPUT_HEIGHT &&
                            orig_width >= 0 && orig_width < INPUT_WIDTH) {
                            pad_value = conv1_out[batch][in_ch][orig_depth][orig_height][orig_width];
                        }
                        cube_buffer2[batch][in_ch][CONV_KERNEL - 1][height][width] = pad_value;
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL - 1; kh++) {
                                    line_buffer2[batch][in_ch][kd][kh][width] =
                                        line_buffer2[batch][in_ch][kd][kh + 1][width];
                                }
                                line_buffer2[batch][in_ch][kd][CONV_KERNEL - 1][width] =
                                    cube_buffer2[batch][in_ch][kd][height][width];
                            }
                        }

                        if (height >= CONV_KERNEL - 1) {
                            // Update window buffer
                            for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                                for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                    for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                        for (int kw = 0; kw < CONV_KERNEL - 1; kw++) {
                                            window_buffer2[batch][in_ch][kd][kh][kw] =
                                                window_buffer2[batch][in_ch][kd][kh][kw + 1];
                                        }
                                        window_buffer2[batch][in_ch][kd][kh][CONV_KERNEL - 1] =
                                            line_buffer2[batch][in_ch][kd][kh][width];
                                    }
                                }
                            }

                            // Convolution computation
                            if (width >= CONV_KERNEL - 1 &&
                                (depth - CONV_KERNEL + 1) % 1 == 0 &&
                                (height - CONV_KERNEL + 1) % 1 == 0 &&
                                (width - CONV_KERNEL + 1) % 1 == 0) {

                                float accum[F_MAP_0];
                                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
                                    #pragma HLS pipeline II=1
                                    accum[out_ch] = 0.0f;

                                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                                        for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                                    float window_val = window_buffer2[batch][in_ch][kd][kh][kw];
                                                    float kernel_val = kernel2[out_ch][in_ch][kd][kh][kw];
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

// Final 1x1 convolution: F_MAP_0 -> OUTPUT_CHANNELS (64 -> 2)
void FinalConv1x1(
    float kernel[OUTPUT_CHANNELS][F_MAP_0][1][1][1],
    float bias[OUTPUT_CHANNELS],
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS array_partition variable=kernel cyclic factor=F_MAP_0 dim=2
    #pragma HLS array_partition variable=bias complete dim=1

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    #pragma HLS pipeline II=1

                    float accum[OUTPUT_CHANNELS];
                    #pragma HLS array_partition variable=accum complete dim=1

                    for (int out_ch = 0; out_ch < OUTPUT_CHANNELS; out_ch++) {
                        accum[out_ch] = bias[out_ch];

                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            accum[out_ch] += input[batch][in_ch][depth][height][width] *
                                           kernel[out_ch][in_ch][0][0][0];
                        }

                        output[batch][out_ch][depth][height][width] = accum[out_ch];
                    }
                }
            }
        }
    }
}

// Sigmoid activation
void Sigmoid3D(
    float input[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS INTERFACE m_axi port=input bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Lookup table for sigmoid values
    // Pre-computed for range [-6, 6] with 0.125 step size (97 entries)
    static const float sigmoid_lut[97] = {
        0.0025f, 0.0033f, 0.0043f, 0.0056f, 0.0074f, 0.0096f, 0.0125f, 0.0163f,
        0.0211f, 0.0273f, 0.0351f, 0.0450f, 0.0574f, 0.0729f, 0.0921f, 0.1157f,
        0.1443f, 0.1788f, 0.2199f, 0.2686f, 0.3258f, 0.3925f, 0.4695f, 0.5000f,
        0.5305f, 0.6075f, 0.6742f, 0.7314f, 0.7801f, 0.8212f, 0.8557f, 0.8843f,
        0.9079f, 0.9271f, 0.9426f, 0.9550f, 0.9649f, 0.9727f, 0.9789f, 0.9837f,
        0.9875f, 0.9904f, 0.9926f, 0.9944f, 0.9957f, 0.9967f, 0.9975f, 0.9981f,
        0.9985f, 0.9988f, 0.9991f, 0.9993f, 0.9995f, 0.9996f, 0.9997f, 0.9998f,
        0.9998f, 0.9999f, 0.9999f, 0.9999f, 0.9999f, 0.9999f, 1.0000f, 1.0000f,
        1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
        1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
        1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
        1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
        1.0000f
    };
    #pragma HLS array_partition variable=sigmoid_lut complete dim=1
    #pragma HLS bind_storage variable=sigmoid_lut type=rom impl=lutram

    const int total_elements = BATCH_SIZE * OUTPUT_CHANNELS * INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH;

    // Flatten all loops for maximum parallelization
    sigmoid_loop: for (int i = 0; i < total_elements; i++) {
        #pragma HLS pipeline II=1
        #pragma HLS unroll factor=8

        // Calculate indices for multi-dimensional access
        int width = i % INPUT_WIDTH;
        int height = (i / INPUT_WIDTH) % INPUT_HEIGHT;
        int depth = (i / (INPUT_WIDTH * INPUT_HEIGHT)) % INPUT_DEPTH;
        int ch = (i / (INPUT_WIDTH * INPUT_HEIGHT * INPUT_DEPTH)) % OUTPUT_CHANNELS;
        int batch = i / (INPUT_WIDTH * INPUT_HEIGHT * INPUT_DEPTH * OUTPUT_CHANNELS);

        float x = input[batch][ch][depth][height][width];

        // Fast sigmoid using lookup table with linear interpolation
        float sigmoid_val;
        if (x <= -6.0f) {
            sigmoid_val = 0.0025f;
        } else if (x >= 6.0f) {
            sigmoid_val = 0.9975f;
        } else {
            // Map x from [-6, 6] to [0, 96] index range
            float scaled_x = (x + 6.0f) * 8.0f;  // 8 = 96/12
            int index = (int)scaled_x;
            float frac = scaled_x - (float)index;

            // Clamp index to valid range
            if (index >= 96) {
                index = 95;
                frac = 1.0f;
            }

            // Linear interpolation between two LUT values
            float y0 = sigmoid_lut[index];
            float y1 = sigmoid_lut[index + 1];
            sigmoid_val = y0 + frac * (y1 - y0);
        }

        output[batch][ch][depth][height][width] = sigmoid_val;
    }
}

// Top-level UNet3D function
void UNet3DReduced(
    // Input
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],

    // Weights for all layers
    float input_conv1_weight[F_MAP_h][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input_conv2_weight[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float encoder_conv1_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float encoder_conv2_weight[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float decoder_conv1_weight[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float decoder_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float output_conv1_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float output_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float final_conv_weight[OUTPUT_CHANNELS][F_MAP_0][1][1][1],
    float final_conv_bias[OUTPUT_CHANNELS],

    // Output
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return
    #pragma HLS interface bram port=input
    #pragma HLS interface bram port=input_conv1_weight
    #pragma HLS interface bram port=input_conv2_weight
    #pragma HLS interface bram port=encoder_conv1_weight
    #pragma HLS interface bram port=encoder_conv2_weight
    #pragma HLS interface bram port=decoder_conv1_weight
    #pragma HLS interface bram port=decoder_conv2_weight
    #pragma HLS interface bram port=output_conv1_weight
    #pragma HLS interface bram port=output_conv2_weight
    #pragma HLS interface bram port=final_conv_weight
    #pragma HLS interface bram port=final_conv_bias
    #pragma HLS interface bram port=output

    #pragma HLS dataflow

    // Intermediate buffers for dataflow
    float input_conv_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=input_conv_out depth=10 type=fifo

    float encoder_pool_out[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_pool_out depth=10 type=fifo

    float encoder_conv_out[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_conv_out depth=10 type=fifo

    float decoder_upsample_out[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_upsample_out depth=10 type=fifo

    float concat_out[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=concat_out depth=10 type=fifo

    float decoder_conv_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_conv_out depth=10 type=fifo

    float output_conv_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=output_conv_out depth=10 type=fifo

    float final_conv_out[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=final_conv_out depth=10 type=fifo

    // UNet forward pass
    InputConv3D(input_conv1_weight, input_conv2_weight, input, input_conv_out);
    EncoderMaxPool3D(input_conv_out, encoder_pool_out);
    EncoderConv3D(encoder_conv1_weight, encoder_conv2_weight, encoder_pool_out, encoder_conv_out);
    DecoderUpsample3D(encoder_conv_out, decoder_upsample_out);
    ConcatenateTensors(input_conv_out, decoder_upsample_out, concat_out);
    DecoderConv3D(decoder_conv1_weight, decoder_conv2_weight, concat_out, decoder_conv_out);
    OutputConv3D(output_conv1_weight, output_conv2_weight, decoder_conv_out, output_conv_out);
    FinalConv1x1(final_conv_weight, final_conv_bias, output_conv_out, final_conv_out);
    Sigmoid3D(final_conv_out, output);
}