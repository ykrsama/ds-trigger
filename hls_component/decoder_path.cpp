#include "include/decoder_path.h"

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

void DecoderConv3D_1(
    float kernel[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
#pragma HLS array_partition variable=kernel cyclic factor=CONCAT_CHANNELS dim=2
#pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=3
#pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=4
#pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=5

    // Buffer definitions for convolution
    float cube_buffer[BATCH_SIZE][CONCAT_CHANNELS][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
#pragma HLS bind_storage variable=cube_buffer type=ram_2p impl=lutram

    float line_buffer[BATCH_SIZE][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
#pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

    float window_buffer[BATCH_SIZE][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
#pragma HLS array_partition variable=window_buffer cyclic factor=CONCAT_CHANNELS dim=2
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
                    for (int in_ch = 0; in_ch < CONCAT_CHANNELS; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer[batch][in_ch][kd][height][width] =
                                cube_buffer[batch][in_ch][kd + 1][height][width];
                        }
                        // Apply padding on the fly
                        float pad_value = 0.0f;
                        int orig_depth = depth - CONV_PADDING;
                        int orig_height = height - CONV_PADDING;
                        int orig_width = width - CONV_PADDING;

                        if (orig_depth >= 0 && orig_depth < INPUT_DEPTH &&
                            orig_height >= 0 && orig_height < INPUT_HEIGHT &&
                            orig_width >= 0 && orig_width < INPUT_WIDTH) {
                            pad_value = input[batch][in_ch][orig_depth][orig_height][orig_width];
                        }
                        cube_buffer[batch][in_ch][CONV_KERNEL - 1][height][width] = pad_value;
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < CONCAT_CHANNELS; in_ch++) {
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
                            for (int in_ch = 0; in_ch < CONCAT_CHANNELS; in_ch++) {
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

                                    for (int in_ch = 0; in_ch < CONCAT_CHANNELS; in_ch++) {
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

void DecoderConv3D_2(
    float kernel[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
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

                        if (orig_depth >= 0 && orig_depth < INPUT_DEPTH &&
                            orig_height >= 0 && orig_height < INPUT_HEIGHT &&
                            orig_width >= 0 && orig_width < INPUT_WIDTH) {
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

void DecoderGroupNorm3D_1(
    float input_data[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[CONCAT_CHANNELS],
    float beta[CONCAT_CHANNELS],
    float output_data[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    int CHANNELS_PER_GROUP = CONCAT_CHANNELS / NUM_GROUPS;
    float N = (float)(INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * CHANNELS_PER_GROUP);

    float gn_buffer[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
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
        for (int ch = 0; ch < CONCAT_CHANNELS; ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
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
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
#pragma HLS pipeline II=1
                    for (int ch = 0; ch < CONCAT_CHANNELS; ch++) {
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

void DecoderGroupNorm3D_2(
    float input_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[F_MAP_0],
    float beta[F_MAP_0],
    float output_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    int CHANNELS_PER_GROUP = F_MAP_0 / NUM_GROUPS;
    float N = (float)(INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * CHANNELS_PER_GROUP);

    float gn_buffer[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
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
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
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
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
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

void DecoderDoubleConv3D(
    float decoder_conv1_weight[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float decoder_conv1_gamma[CONCAT_CHANNELS],
    float decoder_conv1_beta[CONCAT_CHANNELS],
    float decoder_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float decoder_conv2_gamma[F_MAP_0],
    float decoder_conv2_beta[F_MAP_0],
    float input[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
#pragma HLS dataflow

    float gn1_out[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
#pragma HLS stream variable=gn1_out depth=10 type=fifo

    float conv1_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
#pragma HLS stream variable=conv1_out depth=10 type=fifo

    float gn2_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
#pragma HLS stream variable=gn2_out depth=10 type=fifo

    DecoderGroupNorm3D_1(input, decoder_conv1_gamma, decoder_conv1_beta, gn1_out);
    DecoderConv3D_1(decoder_conv1_weight, gn1_out, conv1_out);
    DecoderGroupNorm3D_2(conv1_out, decoder_conv2_gamma, decoder_conv2_beta, gn2_out);
    DecoderConv3D_2(decoder_conv2_weight, gn2_out, output);
}
