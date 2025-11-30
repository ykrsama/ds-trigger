#include "unet3d_reduced.h"

using namespace std;

// Input conv block: 1 -> 64 -> 64 channels
void InputConv3dBlock(
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float conv1_kernel[F_MAPS_0][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv1_gamma[F_MAPS_0],
    float conv1_beta[F_MAPS_0],
    float conv2_kernel[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv2_gamma[F_MAPS_0],
    float conv2_beta[F_MAPS_0],
    float output[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS dataflow

    #pragma HLS array_partition variable=conv1_kernel cyclic factor=8 dim=2
    #pragma HLS array_partition variable=conv1_kernel cyclic factor=3 dim=3
    #pragma HLS array_partition variable=conv1_kernel cyclic factor=3 dim=4
    #pragma HLS array_partition variable=conv1_kernel cyclic factor=3 dim=5

    #pragma HLS array_partition variable=conv2_kernel cyclic factor=8 dim=2
    #pragma HLS array_partition variable=conv2_kernel cyclic factor=3 dim=3
    #pragma HLS array_partition variable=conv2_kernel cyclic factor=3 dim=4
    #pragma HLS array_partition variable=conv2_kernel cyclic factor=3 dim=5

    float padded_input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH+2][INPUT_HEIGHT+2][INPUT_WIDTH+2];
    #pragma HLS stream variable=padded_input depth=10 type=fifo
    #pragma HLS bind_storage variable=padded_input type=ram_t2p impl=bram

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH + 2; depth++) {
            for (int height = 0; height < INPUT_HEIGHT + 2; height++) {
                for (int width = 0; width < INPUT_WIDTH + 2; width++) {
                    for (int ch = 0; ch < INPUT_CHANNELS; ch++) {
                        float pad_value = 0.0f;
                        int orig_depth = depth - PADDING;
                        int orig_height = height - PADDING;
                        int orig_width = width - PADDING;

                        if (orig_depth >= 0 && orig_depth < INPUT_DEPTH &&
                            orig_height >= 0 && orig_height < INPUT_HEIGHT &&
                            orig_width >= 0 && orig_width < INPUT_WIDTH) {
                            pad_value = input[batch][ch][orig_depth][orig_height][orig_width];
                        }
                        padded_input[batch][ch][depth][height][width] = pad_value;
                    }
                }
            }
        }
    }

    float conv1_out[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=uram

    float cube_buffer[BATCH_SIZE][INPUT_CHANNELS][KERNEL_SIZE][INPUT_HEIGHT+2][INPUT_WIDTH+2];
    #pragma HLS bind_storage variable=cube_buffer type=ram_2p impl=lutram

    float line_buffer[BATCH_SIZE][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE][INPUT_WIDTH+2];
    #pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

    float window_buffer[BATCH_SIZE][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
    #pragma HLS array_partition variable=window_buffer cyclic factor=8 dim=2
    #pragma HLS array_partition variable=window_buffer cyclic factor=3 dim=3
    #pragma HLS array_partition variable=window_buffer cyclic factor=3 dim=4
    #pragma HLS array_partition variable=window_buffer cyclic factor=3 dim=5
    #pragma HLS bind_storage variable=window_buffer type=ram_2p impl=lutram

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH + 2; depth++) {
            for (int height = 0; height < INPUT_HEIGHT + 2; height++) {
                for (int width = 0; width < INPUT_WIDTH + 2; width++) {

                    for (int ch = 0; ch < INPUT_CHANNELS; ch++) {
                        for (int kd = 0; kd < KERNEL_SIZE - 1; kd++) {
                            cube_buffer[batch][ch][kd][height][width] =
                                cube_buffer[batch][ch][kd + 1][height][width];
                        }
                        cube_buffer[batch][ch][KERNEL_SIZE - 1][height][width] =
                            padded_input[batch][ch][depth][height][width];
                    }

                    if (depth >= KERNEL_SIZE - 1) {
                        for (int ch = 0; ch < INPUT_CHANNELS; ch++) {
                            for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                                for (int kh = 0; kh < KERNEL_SIZE - 1; kh++) {
                                    line_buffer[batch][ch][kd][kh][width] =
                                        line_buffer[batch][ch][kd][kh + 1][width];
                                }
                                line_buffer[batch][ch][kd][KERNEL_SIZE - 1][width] =
                                    cube_buffer[batch][ch][kd][height][width];
                            }
                        }

                        if (height >= KERNEL_SIZE - 1) {
                            for (int ch = 0; ch < INPUT_CHANNELS; ch++) {
                                for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                                    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                                        for (int kw = 0; kw < KERNEL_SIZE - 1; kw++) {
                                            window_buffer[batch][ch][kd][kh][kw] =
                                                window_buffer[batch][ch][kd][kh][kw + 1];
                                        }
                                        window_buffer[batch][ch][kd][kh][KERNEL_SIZE - 1] =
                                            line_buffer[batch][ch][kd][kh][width];
                                    }
                                }
                            }

                            if (width >= KERNEL_SIZE - 1) {
                                float accum[F_MAPS_0];
                                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < F_MAPS_0; out_ch++) {
                                    #pragma HLS pipeline II=1
                                    accum[out_ch] = 0.0f;

                                    for (int in_ch = 0; in_ch < INPUT_CHANNELS; in_ch++) {
                                        for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                                            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                                                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                                                    float window_val = window_buffer[batch][in_ch][kd][kh][kw];
                                                    float kernel_val = conv1_kernel[out_ch][in_ch][kd][kh][kw];
                                                    accum[out_ch] += window_val * kernel_val;
                                                }
                                            }
                                        }
                                    }

                                    int out_depth = depth - KERNEL_SIZE + 1;
                                    int out_height = height - KERNEL_SIZE + 1;
                                    int out_width = width - KERNEL_SIZE + 1;

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

    float gn_buffer[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=gn_buffer type=ram_2p impl=uram

    int CHANNELS_PER_GROUP = F_MAPS_0 / NUM_GROUPS;
    float N = (float)(INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * CHANNELS_PER_GROUP);

    float group_sum[NUM_GROUPS];
    float group_sq_sum[NUM_GROUPS];
    #pragma HLS array_partition variable=group_sum complete
    #pragma HLS array_partition variable=group_sq_sum complete

    for (int g = 0; g < NUM_GROUPS; g++) {
        #pragma HLS unroll
        group_sum[g] = 0.0f;
        group_sq_sum[g] = 0.0f;
    }

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    for (int ch = 0; ch < F_MAPS_0; ch++) {
                        int group_idx = ch / CHANNELS_PER_GROUP;
                        float value = conv1_out[batch][ch][depth][height][width];
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
        inv_std[g] = 1.0f / sigma;
    }

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    #pragma HLS pipeline II=1
                    for (int ch = 0; ch < F_MAPS_0; ch++) {
                        int group_idx = ch / CHANNELS_PER_GROUP;
                        float value = gn_buffer[batch][ch][depth][height][width];
                        float gamma_param = conv1_gamma[ch];
                        float beta_param = conv1_beta[ch];
                        float group_mean = mean[group_idx];
                        float group_inv_std = inv_std[group_idx];
                        float normalized_value = (value - group_mean) * group_inv_std;
                        float gn_output = normalized_value * gamma_param + beta_param;
                        output[batch][ch][depth][height][width] = gn_output;
                    }
                }
            }
        }
    }

    // Second conv layer: 64 -> 64
    float padded_conv1[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH+2][INPUT_HEIGHT+2][INPUT_WIDTH+2];
    #pragma HLS stream variable=padded_conv1 depth=10 type=fifo
    #pragma HLS bind_storage variable=padded_conv1 type=ram_t2p impl=bram

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH + 2; depth++) {
            for (int height = 0; height < INPUT_HEIGHT + 2; height++) {
                for (int width = 0; width < INPUT_WIDTH + 2; width++) {
                    for (int ch = 0; ch < F_MAPS_0; ch++) {
                        float pad_value = 0.0f;
                        int orig_depth = depth - PADDING;
                        int orig_height = height - PADDING;
                        int orig_width = width - PADDING;

                        if (orig_depth >= 0 && orig_depth < INPUT_DEPTH &&
                            orig_height >= 0 && orig_height < INPUT_HEIGHT &&
                            orig_width >= 0 && orig_width < INPUT_WIDTH) {
                            pad_value = output[batch][ch][orig_depth][orig_height][orig_width];
                        }
                        padded_conv1[batch][ch][depth][height][width] = pad_value;
                    }
                }
            }
        }
    }

    float conv2_out[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=conv2_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv2_out type=ram_t2p impl=uram

    float cube_buffer2[BATCH_SIZE][F_MAPS_0][KERNEL_SIZE][INPUT_HEIGHT+2][INPUT_WIDTH+2];
    #pragma HLS bind_storage variable=cube_buffer2 type=ram_2p impl=lutram

    float line_buffer2[BATCH_SIZE][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][INPUT_WIDTH+2];
    #pragma HLS bind_storage variable=line_buffer2 type=ram_2p impl=lutram

    float window_buffer2[BATCH_SIZE][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=8 dim=2
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=3 dim=3
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=3 dim=4
    #pragma HLS array_partition variable=window_buffer2 cyclic factor=3 dim=5
    #pragma HLS bind_storage variable=window_buffer2 type=ram_2p impl=lutram

    // Process second convolution similar to first
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH + 2; depth++) {
            for (int height = 0; height < INPUT_HEIGHT + 2; height++) {
                for (int width = 0; width < INPUT_WIDTH + 2; width++) {

                    for (int ch = 0; ch < F_MAPS_0; ch++) {
                        for (int kd = 0; kd < KERNEL_SIZE - 1; kd++) {
                            cube_buffer2[batch][ch][kd][height][width] =
                                cube_buffer2[batch][ch][kd + 1][height][width];
                        }
                        cube_buffer2[batch][ch][KERNEL_SIZE - 1][height][width] =
                            padded_conv1[batch][ch][depth][height][width];
                    }

                    if (depth >= KERNEL_SIZE - 1) {
                        for (int ch = 0; ch < F_MAPS_0; ch++) {
                            for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                                for (int kh = 0; kh < KERNEL_SIZE - 1; kh++) {
                                    line_buffer2[batch][ch][kd][kh][width] =
                                        line_buffer2[batch][ch][kd][kh + 1][width];
                                }
                                line_buffer2[batch][ch][kd][KERNEL_SIZE - 1][width] =
                                    cube_buffer2[batch][ch][kd][height][width];
                            }
                        }

                        if (height >= KERNEL_SIZE - 1) {
                            for (int ch = 0; ch < F_MAPS_0; ch++) {
                                for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                                    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                                        for (int kw = 0; kw < KERNEL_SIZE - 1; kw++) {
                                            window_buffer2[batch][ch][kd][kh][kw] =
                                                window_buffer2[batch][ch][kd][kh][kw + 1];
                                        }
                                        window_buffer2[batch][ch][kd][kh][KERNEL_SIZE - 1] =
                                            line_buffer2[batch][ch][kd][kh][width];
                                    }
                                }
                            }

                            if (width >= KERNEL_SIZE - 1) {
                                float accum2[F_MAPS_0];
                                #pragma HLS bind_storage variable=accum2 type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < F_MAPS_0; out_ch++) {
                                    #pragma HLS pipeline II=1
                                    accum2[out_ch] = 0.0f;

                                    for (int in_ch = 0; in_ch < F_MAPS_0; in_ch++) {
                                        for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                                            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                                                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                                                    float window_val = window_buffer2[batch][in_ch][kd][kh][kw];
                                                    float kernel_val = conv2_kernel[out_ch][in_ch][kd][kh][kw];
                                                    accum2[out_ch] += window_val * kernel_val;
                                                }
                                            }
                                        }
                                    }

                                    int out_depth = depth - KERNEL_SIZE + 1;
                                    int out_height = height - KERNEL_SIZE + 1;
                                    int out_width = width - KERNEL_SIZE + 1;

                                    float output_value = accum2[out_ch];
                                    bool is_positive = output_value > 0.0f;
                                    float relu_output = is_positive ? output_value : 0.0f;
                                    conv2_out[batch][out_ch][out_depth][out_height][out_width] = relu_output;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Final GroupNorm and output assignment
    float gn_buffer2[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=gn_buffer2 type=ram_2p impl=uram

    float group_sum2[NUM_GROUPS];
    float group_sq_sum2[NUM_GROUPS];
    #pragma HLS array_partition variable=group_sum2 complete
    #pragma HLS array_partition variable=group_sq_sum2 complete

    for (int g = 0; g < NUM_GROUPS; g++) {
        #pragma HLS unroll
        group_sum2[g] = 0.0f;
        group_sq_sum2[g] = 0.0f;
    }

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    for (int ch = 0; ch < F_MAPS_0; ch++) {
                        int group_idx = ch / CHANNELS_PER_GROUP;
                        float value = conv2_out[batch][ch][depth][height][width];
                        gn_buffer2[batch][ch][depth][height][width] = value;
                        group_sum2[group_idx] += value;
                        group_sq_sum2[group_idx] += (value * value);
                    }
                }
            }
        }
    }

    float mean2[NUM_GROUPS];
    float inv_std2[NUM_GROUPS];
    #pragma HLS array_partition variable=mean2 complete
    #pragma HLS array_partition variable=inv_std2 complete

    for (int g = 0; g < NUM_GROUPS; g++) {
        #pragma HLS unroll
        float mu = group_sum2[g] / N;
        float variance = (group_sq_sum2[g] / N) - (mu * mu);
        float sigma = sqrt(variance + EPSILON);
        mean2[g] = mu;
        inv_std2[g] = 1.0f / sigma;
    }

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    #pragma HLS pipeline II=1
                    for (int ch = 0; ch < F_MAPS_0; ch++) {
                        int group_idx = ch / CHANNELS_PER_GROUP;
                        float value = gn_buffer2[batch][ch][depth][height][width];
                        float gamma_param = conv2_gamma[ch];
                        float beta_param = conv2_beta[ch];
                        float group_mean = mean2[group_idx];
                        float group_inv_std = inv_std2[group_idx];
                        float normalized_value = (value - group_mean) * group_inv_std;
                        float gn_output = normalized_value * gamma_param + beta_param;
                        output[batch][ch][depth][height][width] = gn_output;
                    }
                }
            }
        }
    }
}

// Encoder conv block: 64 -> 128 -> 128 channels for downsampled data
void EncoderConv3dBlock(
    float input[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH/2][INPUT_HEIGHT/2][INPUT_WIDTH/2],
    float conv1_kernel[F_MAPS_1][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv1_gamma[F_MAPS_1],
    float conv1_beta[F_MAPS_1],
    float conv2_kernel[F_MAPS_1][F_MAPS_1][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv2_gamma[F_MAPS_1],
    float conv2_beta[F_MAPS_1],
    float output[BATCH_SIZE][F_MAPS_1][INPUT_DEPTH/2][INPUT_HEIGHT/2][INPUT_WIDTH/2]
) {
    #pragma HLS dataflow

    #pragma HLS array_partition variable=conv1_kernel cyclic factor=8 dim=2
    #pragma HLS array_partition variable=conv1_kernel cyclic factor=3 dim=3
    #pragma HLS array_partition variable=conv1_kernel cyclic factor=3 dim=4
    #pragma HLS array_partition variable=conv1_kernel cyclic factor=3 dim=5

    #pragma HLS array_partition variable=conv2_kernel cyclic factor=8 dim=2
    #pragma HLS array_partition variable=conv2_kernel cyclic factor=3 dim=3
    #pragma HLS array_partition variable=conv2_kernel cyclic factor=3 dim=4
    #pragma HLS array_partition variable=conv2_kernel cyclic factor=3 dim=5

    // Suppress unused parameter warnings for placeholder implementation
    (void)conv1_gamma;
    (void)conv1_beta;
    (void)conv2_gamma;
    (void)conv2_beta;

    float padded_input[BATCH_SIZE][F_MAPS_0][HALF_DEPTH+2][HALF_HEIGHT+2][HALF_WIDTH+2];
    #pragma HLS stream variable=padded_input depth=10 type=fifo
    #pragma HLS bind_storage variable=padded_input type=ram_t2p impl=bram

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < HALF_DEPTH + 2; depth++) {
            for (int height = 0; height < HALF_HEIGHT + 2; height++) {
                for (int width = 0; width < HALF_WIDTH + 2; width++) {
                    for (int ch = 0; ch < F_MAPS_0; ch++) {
                        float pad_value = 0.0f;
                        int orig_depth = depth - PADDING;
                        int orig_height = height - PADDING;
                        int orig_width = width - PADDING;

                        if (orig_depth >= 0 && orig_depth < HALF_DEPTH &&
                            orig_height >= 0 && orig_height < HALF_HEIGHT &&
                            orig_width >= 0 && orig_width < HALF_WIDTH) {
                            pad_value = input[batch][ch][orig_depth][orig_height][orig_width];
                        }
                        padded_input[batch][ch][depth][height][width] = pad_value;
                    }
                }
            }
        }
    }

    float conv1_out[BATCH_SIZE][F_MAPS_1][HALF_DEPTH][HALF_HEIGHT][HALF_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=uram

    // Implementation similar to InputConv3dBlock but with different dimensions
    // [Implementation details omitted for brevity - similar convolution processing]

    // Simplified placeholder for now - full implementation would follow same pattern
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < HALF_DEPTH; depth++) {
            for (int height = 0; height < HALF_HEIGHT; height++) {
                for (int width = 0; width < HALF_WIDTH; width++) {
                    for (int ch = 0; ch < F_MAPS_1; ch++) {
                        output[batch][ch][depth][height][width] = 0.0f; // Placeholder
                    }
                }
            }
        }
    }
}

// Decoder conv block: (64+128) -> 64 -> 64 channels
void DecoderConv3dBlock(
    float input[BATCH_SIZE][F_MAPS_0+F_MAPS_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float conv1_kernel[F_MAPS_0][F_MAPS_0+F_MAPS_1][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv1_gamma[F_MAPS_0],
    float conv1_beta[F_MAPS_0],
    float conv2_kernel[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv2_gamma[F_MAPS_0],
    float conv2_beta[F_MAPS_0],
    float output[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS dataflow

    // Suppress unused parameter warnings for placeholder implementation
    (void)input;
    (void)conv1_kernel;
    (void)conv1_gamma;
    (void)conv1_beta;
    (void)conv2_kernel;
    (void)conv2_gamma;
    (void)conv2_beta;

    // Simplified placeholder for now
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    for (int ch = 0; ch < F_MAPS_0; ch++) {
                        output[batch][ch][depth][height][width] = 0.0f; // Placeholder
                    }
                }
            }
        }
    }
}

// Output conv block: 64 -> 64 -> 64 channels
void OutputConv3dBlock(
    float input[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float conv1_kernel[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv1_gamma[F_MAPS_0],
    float conv1_beta[F_MAPS_0],
    float conv2_kernel[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float conv2_gamma[F_MAPS_0],
    float conv2_beta[F_MAPS_0],
    float output[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS dataflow

    // Suppress unused parameter warnings for placeholder implementation
    (void)input;
    (void)conv1_kernel;
    (void)conv1_gamma;
    (void)conv1_beta;
    (void)conv2_kernel;
    (void)conv2_gamma;
    (void)conv2_beta;

    // Simplified placeholder for now
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    for (int ch = 0; ch < F_MAPS_0; ch++) {
                        output[batch][ch][depth][height][width] = 0.0f; // Placeholder
                    }
                }
            }
        }
    }
}

void MaxPool3d(
    float input[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH/2][INPUT_HEIGHT/2][INPUT_WIDTH/2]
) {
    #pragma HLS dataflow

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int ch = 0; ch < F_MAPS_0; ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth += POOL_STRIDE) {
                for (int height = 0; height < INPUT_HEIGHT; height += POOL_STRIDE) {
                    for (int width = 0; width < INPUT_WIDTH; width += POOL_STRIDE) {
                        #pragma HLS pipeline II=1

                        float max_val = -INFINITY;
                        for (int pd = 0; pd < POOL_SIZE; pd++) {
                            for (int ph = 0; ph < POOL_SIZE; ph++) {
                                for (int pw = 0; pw < POOL_SIZE; pw++) {
                                    int d_idx = depth + pd;
                                    int h_idx = height + ph;
                                    int w_idx = width + pw;

                                    if (d_idx < INPUT_DEPTH && h_idx < INPUT_HEIGHT && w_idx < INPUT_WIDTH) {
                                        float val = input[batch][ch][d_idx][h_idx][w_idx];
                                        max_val = (val > max_val) ? val : max_val;
                                    }
                                }
                            }
                        }

                        int out_d = depth / POOL_STRIDE;
                        int out_h = height / POOL_STRIDE;
                        int out_w = width / POOL_STRIDE;

                        if (out_d < INPUT_DEPTH/2 && out_h < INPUT_HEIGHT/2 && out_w < INPUT_WIDTH/2) {
                            output[batch][ch][out_d][out_h][out_w] = max_val;
                        }
                    }
                }
            }
        }
    }
}

void NearestUpsample3d(
    float input[BATCH_SIZE][F_MAPS_1][INPUT_DEPTH/2][INPUT_HEIGHT/2][INPUT_WIDTH/2],
    float output[BATCH_SIZE][F_MAPS_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS dataflow

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int ch = 0; ch < F_MAPS_1; ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
                        #pragma HLS pipeline II=1

                        int src_d = depth / 2;
                        int src_h = height / 2;
                        int src_w = width / 2;

                        if (src_d < INPUT_DEPTH/2 && src_h < INPUT_HEIGHT/2 && src_w < INPUT_WIDTH/2) {
                            output[batch][ch][depth][height][width] = input[batch][ch][src_d][src_h][src_w];
                        } else {
                            output[batch][ch][depth][height][width] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void Concatenate3d(
    float input1[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float input2[BATCH_SIZE][F_MAPS_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAPS_0+F_MAPS_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS dataflow

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    #pragma HLS pipeline II=1

                    for (int ch = 0; ch < F_MAPS_0; ch++) {
                        output[batch][ch][depth][height][width] = input1[batch][ch][depth][height][width];
                    }

                    for (int ch = 0; ch < F_MAPS_1; ch++) {
                        output[batch][F_MAPS_0 + ch][depth][height][width] = input2[batch][ch][depth][height][width];
                    }
                }
            }
        }
    }
}

void unet3d_reduced(
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],

    float input_conv1_weight[F_MAPS_0][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float input_conv1_gamma[F_MAPS_0],
    float input_conv1_beta[F_MAPS_0],
    float input_conv2_weight[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float input_conv2_gamma[F_MAPS_0],
    float input_conv2_beta[F_MAPS_0],

    float encoder_conv1_weight[F_MAPS_1][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float encoder_conv1_gamma[F_MAPS_1],
    float encoder_conv1_beta[F_MAPS_1],
    float encoder_conv2_weight[F_MAPS_1][F_MAPS_1][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float encoder_conv2_gamma[F_MAPS_1],
    float encoder_conv2_beta[F_MAPS_1],

    float decoder_conv1_weight[F_MAPS_0][F_MAPS_0+F_MAPS_1][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float decoder_conv1_gamma[F_MAPS_0],
    float decoder_conv1_beta[F_MAPS_0],
    float decoder_conv2_weight[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float decoder_conv2_gamma[F_MAPS_0],
    float decoder_conv2_beta[F_MAPS_0],

    float output_conv1_weight[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float output_conv1_gamma[F_MAPS_0],
    float output_conv1_beta[F_MAPS_0],
    float output_conv2_weight[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE],
    float output_conv2_gamma[F_MAPS_0],
    float output_conv2_beta[F_MAPS_0],

    float final_conv_weight[OUTPUT_CHANNELS][F_MAPS_0][1][1][1],
    float final_conv_bias[OUTPUT_CHANNELS],

    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return
    #pragma HLS interface bram port=input
    #pragma HLS bind_storage variable=input type=ram_t2p impl=bram
    #pragma HLS interface bram port=output
    #pragma HLS bind_storage variable=output type=ram_t2p impl=bram

    #pragma HLS dataflow

    float f_maps_0_skip[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=f_maps_0_skip depth=10 type=fifo

    float f_maps_0_pooled[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH/2][INPUT_HEIGHT/2][INPUT_WIDTH/2];
    #pragma HLS stream variable=f_maps_0_pooled depth=10 type=fifo

    float f_maps_1[BATCH_SIZE][F_MAPS_1][INPUT_DEPTH/2][INPUT_HEIGHT/2][INPUT_WIDTH/2];
    #pragma HLS stream variable=f_maps_1 depth=10 type=fifo

    float f_maps_1_upsampled[BATCH_SIZE][F_MAPS_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=f_maps_1_upsampled depth=10 type=fifo

    float concatenated[BATCH_SIZE][F_MAPS_0+F_MAPS_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=concatenated depth=10 type=fifo

    float decoder_out[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_out depth=10 type=fifo

    float output_conv_out[BATCH_SIZE][F_MAPS_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=output_conv_out depth=10 type=fifo

    InputConv3dBlock(
        input, input_conv1_weight, input_conv1_gamma, input_conv1_beta,
        input_conv2_weight, input_conv2_gamma, input_conv2_beta,
        f_maps_0_skip
    );

    MaxPool3d(f_maps_0_skip, f_maps_0_pooled);

    EncoderConv3dBlock(
        f_maps_0_pooled, encoder_conv1_weight, encoder_conv1_gamma, encoder_conv1_beta,
        encoder_conv2_weight, encoder_conv2_gamma, encoder_conv2_beta,
        f_maps_1
    );

    NearestUpsample3d(f_maps_1, f_maps_1_upsampled);

    Concatenate3d(f_maps_0_skip, f_maps_1_upsampled, concatenated);

    DecoderConv3dBlock(
        concatenated, decoder_conv1_weight, decoder_conv1_gamma, decoder_conv1_beta,
        decoder_conv2_weight, decoder_conv2_gamma, decoder_conv2_beta,
        decoder_out
    );

    OutputConv3dBlock(
        decoder_out, output_conv1_weight, output_conv1_gamma, output_conv1_beta,
        output_conv2_weight, output_conv2_gamma, output_conv2_beta,
        output_conv_out
    );

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    #pragma HLS pipeline II=1
                    for (int out_ch = 0; out_ch < OUTPUT_CHANNELS; out_ch++) {
                        float sum = final_conv_bias[out_ch];
                        for (int in_ch = 0; in_ch < F_MAPS_0; in_ch++) {
                            sum += output_conv_out[batch][in_ch][depth][height][width] *
                                   final_conv_weight[out_ch][in_ch][0][0][0];
                        }
                        output[batch][out_ch][depth][height][width] = sum;
                    }
                }
            }
        }
    }
}