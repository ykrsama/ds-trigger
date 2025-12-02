#include "include/encoder_path.h"

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

void EncoderConv3D_1(
    float kernel[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
) {
#pragma HLS array_partition variable=kernel cyclic factor=F_MAP_0 dim=2
#pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=3
#pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=4
#pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=5

    // Buffer definitions for convolution
    float cube_buffer[BATCH_SIZE][F_MAP_0][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
#pragma HLS bind_storage variable=cube_buffer type=ram_2p impl=lutram

    float line_buffer[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
#pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

    float window_buffer[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
#pragma HLS array_partition variable=window_buffer cyclic factor=F_MAP_0 dim=2
#pragma HLS array_partition variable=window_buffer cyclic factor=CONV_KERNEL dim=3
#pragma HLS array_partition variable=window_buffer cyclic factor=CONV_KERNEL dim=4
#pragma HLS array_partition variable=window_buffer cyclic factor=CONV_KERNEL dim=5
#pragma HLS bind_storage variable=window_buffer type=ram_2p impl=lutram

    // Convolution computation with padding
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {

                    // Update cube buffer with padded input
                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer[batch][in_ch][kd][height][width] =
                                cube_buffer[batch][in_ch][kd + 1][height][width];
                        }
                        // Apply padding on the fly
                        float pad_value = 0.0f;
                        int orig_depth = depth - CONV_PADDING;
                        int orig_height = height - CONV_PADDING;
                        int orig_width = width - CONV_PADDING;

                        if (orig_depth >= 0 && orig_depth < POOL_OUTPUT_DEPTH &&
                            orig_height >= 0 && orig_height < POOL_OUTPUT_HEIGHT &&
                            orig_width >= 0 && orig_width < POOL_OUTPUT_WIDTH) {
                            pad_value = input[batch][in_ch][orig_depth][orig_height][orig_width];
                        }
                        cube_buffer[batch][in_ch][CONV_KERNEL - 1][height][width] = pad_value;
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
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
                            for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
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

                                float accum[F_MAP_0];
#pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
#pragma HLS pipeline II=1
                                    accum[out_ch] = 0.0f;

                                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
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

void EncoderConv3D_2(
    float kernel[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
) {
#pragma HLS array_partition variable=kernel cyclic factor=F_MAP_0 dim=2
#pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=3
#pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=4
#pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=5

    // Buffer definitions for convolution
    float cube_buffer[BATCH_SIZE][F_MAP_0][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
#pragma HLS bind_storage variable=cube_buffer type=ram_2p impl=lutram

    float line_buffer[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
#pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

    float window_buffer[BATCH_SIZE][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
#pragma HLS array_partition variable=window_buffer cyclic factor=F_MAP_0 dim=2
#pragma HLS array_partition variable=window_buffer cyclic factor=CONV_KERNEL dim=3
#pragma HLS array_partition variable=window_buffer cyclic factor=CONV_KERNEL dim=4
#pragma HLS array_partition variable=window_buffer cyclic factor=CONV_KERNEL dim=5
#pragma HLS bind_storage variable=window_buffer type=ram_2p impl=lutram

    // Convolution computation with padding
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {

                    // Update cube buffer with padded input
                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer[batch][in_ch][kd][height][width] =
                                cube_buffer[batch][in_ch][kd + 1][height][width];
                        }
                        // Apply padding on the fly
                        float pad_value = 0.0f;
                        int orig_depth = depth - CONV_PADDING;
                        int orig_height = height - CONV_PADDING;
                        int orig_width = width - CONV_PADDING;

                        if (orig_depth >= 0 && orig_depth < POOL_OUTPUT_DEPTH &&
                            orig_height >= 0 && orig_height < POOL_OUTPUT_HEIGHT &&
                            orig_width >= 0 && orig_width < POOL_OUTPUT_WIDTH) {
                            pad_value = input[batch][in_ch][orig_depth][orig_height][orig_width];
                        }
                        cube_buffer[batch][in_ch][CONV_KERNEL - 1][height][width] = pad_value;
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
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
                            for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
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

                                float accum[F_MAP_1];
#pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < F_MAP_1; out_ch++) {
#pragma HLS pipeline II=1
                                    accum[out_ch] = 0.0f;

                                    for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
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

void EncoderGroupNorm3D_1(
    float input_data[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    float gamma[F_MAP_0],
    float beta[F_MAP_0],
    float output_data[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
) {
    int CHANNELS_PER_GROUP = F_MAP_0 / NUM_GROUPS;
    float N = (float)(POOL_OUTPUT_DEPTH * POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * CHANNELS_PER_GROUP);

    float gn_buffer[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
#pragma HLS bind_storage variable=gn_buffer type=ram_2p impl=uram

    float group_sum[NUM_GROUPS];
    float group_sq_sum[NUM_GROUPS];
#pragma HLS array_partition variable=group_sum complete
#pragma HLS array_partition variable=group_sq_sum complete

    for (int g = 0; g < NUM_GROUPS; g++) {
#pragma HLS unroll
        group_sum[g] = (float)0.000000;
        group_sq_sum[g] = (float)0.000000;
    }

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int ch = 0; ch < F_MAP_0; ch++) {
            for (int depth = 0; depth < POOL_OUTPUT_DEPTH; depth++) {
                for (int height = 0; height < POOL_OUTPUT_HEIGHT; height++) {
                    for (int width = 0; width < POOL_OUTPUT_WIDTH; width++) {
                        int group_idx = ch / CHANNELS_PER_GROUP;

                        float value = input_data[batch][ch][depth][height][width];

                        gn_buffer[batch][ch][depth][height][width] = value;

                        group_sum[group_idx] += value;
                        group_sq_sum[group_idx] += (value * value);
                    }
                }
            }
        }
    }

    float mean[NUM_GROUPS];
    float inv_std[NUM_GROUPS];
#pragma HLS array_partition variable=mean complete
#pragma HLS array_partition variable=inv_std complete

    for (int g = 0; g < NUM_GROUPS; g++) {
#pragma HLS unroll
        float mu = group_sum[g] / N;
        float variance = (group_sq_sum[g] / N) - (mu * mu);
        float sigma = sqrt(variance + EPSILON);

        mean[g] = mu;
        inv_std[g] = float(1) / sigma;
    }

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < POOL_OUTPUT_DEPTH; depth++) {
            for (int height = 0; height < POOL_OUTPUT_HEIGHT; height++) {
                for (int width = 0; width < POOL_OUTPUT_WIDTH; width++) {
#pragma HLS pipeline II=1
                    for (int ch = 0; ch < F_MAP_0; ch++) {
                        int group_idx = ch / CHANNELS_PER_GROUP;

                        float value = gn_buffer[batch][ch][depth][height][width];

                        float gamma_param = gamma[ch];
                        float beta_param = beta[ch];

                        float group_mean = mean[group_idx];
                        float group_inv_std = inv_std[group_idx];

                        float normalized_value = (value - group_mean) * group_inv_std;

                        float output_value = normalized_value * gamma_param + beta_param;

                        output_data[batch][ch][depth][height][width] = output_value;
                    }
                }
            }
        }
    }
}

void EncoderGroupNorm3D_2(
    float input_data[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    float gamma[F_MAP_0],
    float beta[F_MAP_0],
    float output_data[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
) {
    int CHANNELS_PER_GROUP = F_MAP_0 / NUM_GROUPS;
    float N = (float)(POOL_OUTPUT_DEPTH * POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * CHANNELS_PER_GROUP);

    float gn_buffer[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
#pragma HLS bind_storage variable=gn_buffer type=ram_2p impl=uram

    float group_sum[NUM_GROUPS];
    float group_sq_sum[NUM_GROUPS];
#pragma HLS array_partition variable=group_sum complete
#pragma HLS array_partition variable=group_sq_sum complete

    for (int g = 0; g < NUM_GROUPS; g++) {
#pragma HLS unroll
        group_sum[g] = (float)0.000000;
        group_sq_sum[g] = (float)0.000000;
    }

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int ch = 0; ch < F_MAP_0; ch++) {
            for (int depth = 0; depth < POOL_OUTPUT_DEPTH; depth++) {
                for (int height = 0; height < POOL_OUTPUT_HEIGHT; height++) {
                    for (int width = 0; width < POOL_OUTPUT_WIDTH; width++) {
                        int group_idx = ch / CHANNELS_PER_GROUP;

                        float value = input_data[batch][ch][depth][height][width];

                        gn_buffer[batch][ch][depth][height][width] = value;

                        group_sum[group_idx] += value;
                        group_sq_sum[group_idx] += (value * value);
                    }
                }
            }
        }
    }

    float mean[NUM_GROUPS];
    float inv_std[NUM_GROUPS];
#pragma HLS array_partition variable=mean complete
#pragma HLS array_partition variable=inv_std complete

    for (int g = 0; g < NUM_GROUPS; g++) {
#pragma HLS unroll
        float mu = group_sum[g] / N;
        float variance = (group_sq_sum[g] / N) - (mu * mu);
        float sigma = sqrt(variance + EPSILON);

        mean[g] = mu;
        inv_std[g] = float(1) / sigma;
    }

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < POOL_OUTPUT_DEPTH; depth++) {
            for (int height = 0; height < POOL_OUTPUT_HEIGHT; height++) {
                for (int width = 0; width < POOL_OUTPUT_WIDTH; width++) {
#pragma HLS pipeline II=1
                    for (int ch = 0; ch < F_MAP_0; ch++) {
                        int group_idx = ch / CHANNELS_PER_GROUP;

                        float value = gn_buffer[batch][ch][depth][height][width];

                        float gamma_param = gamma[ch];
                        float beta_param = beta[ch];

                        float group_mean = mean[group_idx];
                        float group_inv_std = inv_std[group_idx];

                        float normalized_value = (value - group_mean) * group_inv_std;

                        float output_value = normalized_value * gamma_param + beta_param;

                        output_data[batch][ch][depth][height][width] = output_value;
                    }
                }
            }
        }
    }
}

void EncoderDoubleConv3D(
    float encoder_conv1_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float encoder_conv1_gamma[F_MAP_0],
    float encoder_conv1_beta[F_MAP_0],
    float encoder_conv2_weight[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float encoder_conv2_gamma[F_MAP_0],
    float encoder_conv2_beta[F_MAP_0],
    float input[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
) {
    float gn1_out[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
#pragma HLS stream variable=gn1_out depth=10 type=fifo

    float conv1_out[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
#pragma HLS stream variable=conv1_out depth=10 type=fifo

    float gn2_out[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
#pragma HLS stream variable=gn2_out depth=10 type=fifo

    EncoderGroupNorm3D_1(input, encoder_conv1_gamma, encoder_conv1_beta, gn1_out);
    EncoderConv3D_1(encoder_conv1_weight, gn1_out, conv1_out);
    EncoderGroupNorm3D_2(conv1_out, encoder_conv2_gamma, encoder_conv2_beta, gn2_out);
    EncoderConv3D_2(encoder_conv2_weight, gn2_out, output);
}
