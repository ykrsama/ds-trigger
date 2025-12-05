#include "../template/model.h"


// GroupNorm3D template
template<int T_IN_CHANNELS,
         int T_INPUT_DEPTH,
         int T_INPUT_HEIGHT,
         int T_INPUT_WIDTH>
void GroupNorm3D(float input_data[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                 float gamma[T_IN_CHANNELS],
                 float beta[T_IN_CHANNELS],
                 float output_data[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH]) {
    // calculate channel number
    int CHANNELS_PER_GROUP = T_IN_CHANNELS / NUM_GROUPS;
    float N = (float) (T_INPUT_DEPTH * T_INPUT_HEIGHT * T_INPUT_WIDTH * CHANNELS_PER_GROUP);

    // 1. On chip buffer
    float gn_buffer[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH];
    #pragma HLS bind_storage variable=gn_buffer type=ram_2p impl=uram

    // Statistics
    float group_sum[NUM_GROUPS];
    float group_sq_sum[NUM_GROUPS];
    #pragma HLS array_partition variable=group_sum complete
    #pragma HLS array_partition variable=group_sq_sum complete

    for (int g = 0; g < NUM_GROUPS; g++) {
    #pragma HLS unroll
        group_sum[g] = (float) 0.000000;
        group_sq_sum[g] = (float) 0.000000;
    }

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int ch = 0; ch < T_IN_CHANNELS; ch++) {
            for (int depth = 0; depth < T_INPUT_DEPTH; depth++) {
                for (int height = 0; height < T_INPUT_HEIGHT; height++) {
                    for (int width = 0; width < T_INPUT_WIDTH; width++) {
                        // channel group id
                        int group_idx = ch / CHANNELS_PER_GROUP;
                        // read input data
                        float value = input_data[batch][ch][depth][height][width];
                        // write buffer
                        gn_buffer[batch][ch][depth][height][width] = value;
                        // add statistics
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
        for (int depth = 0; depth < T_INPUT_DEPTH; depth++) {
            for (int height = 0; height < T_INPUT_HEIGHT; height++) {
                for (int width = 0; width < T_INPUT_WIDTH; width++) {
                    #pragma HLS pipeline II=1
                    for (int ch = 0; ch < T_IN_CHANNELS; ch++) {
                        // channel group id
                        int group_idx = ch / CHANNELS_PER_GROUP;
                        // read from buffer
                        float value = gn_buffer[batch][ch][depth][height][width];
                        // gamma and beta
                        float gamma_param = gamma[ch];
                        float beta_param = beta[ch];
                        // statistics
                        float group_mean = mean[group_idx];
                        float group_inv_std = inv_std[group_idx];
                        // normalize
                        float normalized_value = (value - group_mean) * group_inv_std;
                        // scale and shift
                        float output_value = normalized_value * gamma_param + beta_param;
                        // write output
                        output_data[batch][ch][depth][height][width] = output_value;
                    }
                }
            }
        }
    }
}

// Conv3D template
template<int T_IN_CHANNELS,
         int T_OUT_CHANNELS,
         int T_INPUT_DEPTH,
         int T_INPUT_HEIGHT,
         int T_INPUT_WIDTH>
void Conv3D(float kernel[T_OUT_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
            float input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
            float output[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH]) {
    #pragma HLS array_partition variable=kernel complete dim=2
    #pragma HLS array_partition variable=kernel complete dim=3
    #pragma HLS array_partition variable=kernel complete dim=4
    #pragma HLS array_partition variable=kernel complete dim=5
    #pragma HLS array_partition variable=input cyclic factor=T_IN_CHANNELS dim=2
    #pragma HLS array_partition variable=output cyclic factor=T_OUT_CHANNELS dim=2

    // padded size
    const int PADDED_DEPTH = T_INPUT_DEPTH + 2 * CONV_PADDING;
    const int PADDED_HEIGHT = T_INPUT_HEIGHT + 2 * CONV_PADDING;
    const int PADDED_WIDTH = T_INPUT_WIDTH + 2 * CONV_PADDING;

    // fill input
    float padded_input[BATCH_SIZE][T_IN_CHANNELS][PADDED_DEPTH][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS array_partition variable=padded_input cyclic factor=T_IN_CHANNELS dim=2
    #pragma HLS bind_storage variable=padded_input type=ram_t2p impl=bram

    // Filling operation with better memory access patterns
    PaddingBatch: for (int batch = 0; batch < BATCH_SIZE; batch++) {
        PaddingDepth: for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            PaddingHeight: for (int height = 0; height < PADDED_HEIGHT; height++) {
                PaddingWidth: for (int width = 0; width < PADDED_WIDTH; width++) {
                    #pragma HLS pipeline II=1
                    PaddingChan: for (int in_ch = 0; in_ch < T_IN_CHANNELS; in_ch++) {
                        #pragma HLS unroll factor=2
                        float pad_value = (float) 0.000000;
                        int orig_depth = depth - CONV_PADDING;
                        int orig_height = height - CONV_PADDING;
                        int orig_width = width - CONV_PADDING;

                        bool valid_pixel = (orig_depth >= 0 && orig_depth < T_INPUT_DEPTH &&
                                          orig_height >= 0 && orig_height < T_INPUT_HEIGHT &&
                                          orig_width >= 0 && orig_width < T_INPUT_WIDTH);

                        pad_value = valid_pixel ? input[batch][in_ch][orig_depth][orig_height][orig_width] : (float)0.000000;
                        padded_input[batch][in_ch][depth][height][width] = pad_value;
                    }
                }
            }
        }
    }

    // Buffer definitions for convolution - optimized memory interfaces
    float cube_buffer[BATCH_SIZE][T_IN_CHANNELS][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS array_partition variable=cube_buffer cyclic factor=T_IN_CHANNELS dim=2
    #pragma HLS array_partition variable=cube_buffer complete dim=3
    #pragma HLS bind_storage variable=cube_buffer type=ram_2p impl=lutram

    float line_buffer[BATCH_SIZE][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
    #pragma HLS array_partition variable=line_buffer cyclic factor=T_IN_CHANNELS dim=2
    #pragma HLS array_partition variable=line_buffer complete dim=3
    #pragma HLS array_partition variable=line_buffer complete dim=4
    #pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

    float window_buffer[BATCH_SIZE][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    #pragma HLS array_partition variable=window_buffer complete dim=2
    #pragma HLS array_partition variable=window_buffer complete dim=3
    #pragma HLS array_partition variable=window_buffer complete dim=4
    #pragma HLS array_partition variable=window_buffer complete dim=5
    #pragma HLS bind_storage variable=window_buffer type=ram_2p impl=lutram

    // Convolution computation with enhanced parallelization
    ConvBatch: for (int batch = 0; batch < BATCH_SIZE; batch++) {
        ConvDepth: for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            ConvHeight: for (int height = 0; height < PADDED_HEIGHT; height++) {
                ConvWidth: for (int width = 0; width < PADDED_WIDTH; width++) {

                    // Update cube buffer - parallel channel processing
                    UpdateCubeBuffer: for (int in_ch = 0; in_ch < T_IN_CHANNELS; in_ch++) {
                        #pragma HLS pipeline II=1
                        #pragma HLS unroll factor=2
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            #pragma HLS unroll
                            cube_buffer[batch][in_ch][kd][height][width] =
                                cube_buffer[batch][in_ch][kd + 1][height][width];
                        }
                        cube_buffer[batch][in_ch][CONV_KERNEL - 1][height][width] =
                            padded_input[batch][in_ch][depth][height][width];
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer - parallel processing
                        UpdateLineBuffer: for (int in_ch = 0; in_ch < T_IN_CHANNELS; in_ch++) {
                            #pragma HLS pipeline II=1
                            #pragma HLS unroll factor=2
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                #pragma HLS unroll
                                for (int kh = 0; kh < CONV_KERNEL - 1; kh++) {
                                    #pragma HLS unroll
                                    line_buffer[batch][in_ch][kd][kh][width] =
                                        line_buffer[batch][in_ch][kd][kh + 1][width];
                                }
                                line_buffer[batch][in_ch][kd][CONV_KERNEL - 1][width] =
                                    cube_buffer[batch][in_ch][kd][height][width];
                            }
                        }

                        if (height >= CONV_KERNEL - 1) {
                            // Update window buffer
                            UpdateWindowBuffer: for (int in_ch = 0; in_ch < T_IN_CHANNELS; in_ch++) {
                                #pragma HLS pipeline II=1
                                #pragma HLS unroll factor=2
                                for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                    #pragma HLS unroll
                                    for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                        #pragma HLS unroll
                                        for (int kw = 0; kw < CONV_KERNEL - 1; kw++) {
                                            #pragma HLS unroll
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
                                (depth) % CONV_STRIDE == 0 &&
                                (height) % CONV_STRIDE == 0 &&
                                (width) % CONV_STRIDE == 0) {

                                float accum[T_OUT_CHANNELS];
                                #pragma HLS array_partition variable=accum complete dim=1
                                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                ConvKernelReLU: for (int out_ch = 0; out_ch < T_OUT_CHANNELS; out_ch++) {
                                    #pragma HLS pipeline II=1
                                    accum[out_ch] = (float)0.000000;

                                    for (int in_ch = 0; in_ch < T_IN_CHANNELS; in_ch++) {
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

                                    int out_depth = (depth - CONV_KERNEL) / CONV_STRIDE + 1;
                                    int out_height = (height - CONV_KERNEL) / CONV_STRIDE + 1;
                                    int out_width = (width - CONV_KERNEL) / CONV_STRIDE + 1;

                                    // ReLU activation
                                    float output_value = accum[out_ch];
                                    bool is_positive = output_value > (float)0.000000;
                                    float relu_output = is_positive ? output_value : (float)0.000000;
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

template<int T_IN_CHANNELS,
         int T_MID_CHANNELS,
         int T_OUT_CHANNELS,
         int T_INPUT_DEPTH,
         int T_INPUT_HEIGHT,
         int T_INPUT_WIDTH>
void DoubleConv3D(float input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                  float kernel1[T_MID_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
                  float gamma1[T_IN_CHANNELS],
                  float beta1[T_IN_CHANNELS],
                  float kernel2[T_OUT_CHANNELS][T_MID_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
                  float gamma2[T_MID_CHANNELS],
                  float beta2[T_MID_CHANNELS],
                  float output[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH]) {
    // buffers
    float gn1_out[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH];
    #pragma HLS stream variable=gn1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=gn1_out type=ram_t2p impl=bram

    float conv1_out[BATCH_SIZE][T_MID_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=bram

    float gn2_out[BATCH_SIZE][T_MID_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH];
    #pragma HLS stream variable=gn2_out depth=10 type=fifo
    #pragma HLS bind_storage variable=gn2_out type=ram_t2p impl=bram

    GroupNorm3D<T_IN_CHANNELS, T_INPUT_DEPTH, T_INPUT_HEIGHT, T_INPUT_WIDTH>(input, gamma1, beta1, gn1_out);
    Conv3D<T_IN_CHANNELS, T_MID_CHANNELS, T_INPUT_DEPTH, T_INPUT_HEIGHT, T_INPUT_WIDTH>(kernel1, gn1_out, conv1_out);
    GroupNorm3D<T_MID_CHANNELS, T_INPUT_DEPTH, T_INPUT_HEIGHT, T_INPUT_WIDTH>(conv1_out, gamma2, beta2, gn2_out);
    Conv3D<T_MID_CHANNELS, T_OUT_CHANNELS, T_INPUT_DEPTH, T_INPUT_HEIGHT, T_INPUT_WIDTH>(kernel2, gn2_out, output);
}

template<int T_IN_CHANNELS,
         int T_MID_CHANNELS,
         int T_OUT_CHANNELS,
         int T_INPUT_DEPTH,
         int T_INPUT_HEIGHT,
         int T_INPUT_WIDTH>
void DoubleConv3D2Head(float input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                       float kernel1[T_MID_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
                       float gamma1[T_IN_CHANNELS],
                       float beta1[T_IN_CHANNELS],
                       float kernel2[T_OUT_CHANNELS][T_MID_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
                       float gamma2[T_MID_CHANNELS],
                       float beta2[T_MID_CHANNELS],
                       float output1[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                       float output2[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH]) {
    // buffers
    float gn1_out[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH];
    #pragma HLS stream variable=gn1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=gn1_out type=ram_t2p impl=bram

    float conv1_out[BATCH_SIZE][T_MID_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=bram

    float gn2_out[BATCH_SIZE][T_MID_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH];
    #pragma HLS stream variable=gn2_out depth=10 type=fifo
    #pragma HLS bind_storage variable=gn2_out type=ram_t2p impl=bram

    // Temporary output buffer
    float temp_output[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_DEPTH][T_INPUT_HEIGHT][T_INPUT_WIDTH];
    #pragma HLS stream variable=temp_output depth=10 type=fifo
    #pragma HLS bind_storage variable=temp_output type=ram_t2p impl=bram

    GroupNorm3D<T_IN_CHANNELS>(input, gamma1, beta1, gn1_out);
    Conv3D<T_IN_CHANNELS, T_MID_CHANNELS>(kernel1, gn1_out, conv1_out);
    GroupNorm3D<T_MID_CHANNELS>(conv1_out, gamma2, beta2, gn2_out);
    Conv3D<T_MID_CHANNELS, T_OUT_CHANNELS>(kernel2, gn2_out, temp_output);

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

template<int T_IN_CHANNELS>
void MaxPool3D(float input[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
               float output[BATCH_SIZE][T_IN_CHANNELS][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]) {
    // Buffer definitions
    float cube_buffer[BATCH_SIZE][T_IN_CHANNELS][POOL_KERNEL][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=cube_buffer type=ram_2p impl=lutram

    float line_buffer[BATCH_SIZE][T_IN_CHANNELS][POOL_KERNEL][POOL_KERNEL][INPUT_WIDTH];
    #pragma HLS array_partition variable=line_buffer cyclic factor=T_IN_CHANNELS dim=2
    #pragma HLS array_partition variable=line_buffer complete dim=3
    #pragma HLS array_partition variable=line_buffer complete dim=4
    #pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

    float window_buffer[BATCH_SIZE][T_IN_CHANNELS][POOL_KERNEL][POOL_KERNEL][POOL_KERNEL];
    #pragma HLS array_partition variable=window_buffer cyclic factor=T_IN_CHANNELS dim=2
    #pragma HLS array_partition variable=window_buffer cyclic factor=POOL_KERNEL dim=3
    #pragma HLS array_partition variable=window_buffer cyclic factor=POOL_KERNEL dim=4
    #pragma HLS array_partition variable=window_buffer cyclic factor=POOL_KERNEL dim=5
    #pragma HLS bind_storage variable=window_buffer type=ram_2p impl=lutram

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {

                    // Update cube buffer
                    for (int in_ch = 0; in_ch < T_IN_CHANNELS; in_ch++) {
                        for (int kd = 0; kd < POOL_KERNEL - 1; kd++) {
                            cube_buffer[batch][in_ch][kd][height][width] =
                                cube_buffer[batch][in_ch][kd + 1][height][width];
                        }
                        cube_buffer[batch][in_ch][POOL_KERNEL - 1][height][width] =
                            input[batch][in_ch][depth][height][width];
                    }

                    if (depth >= POOL_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < T_IN_CHANNELS; in_ch++) {
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
                            for (int in_ch = 0; in_ch < T_IN_CHANNELS; in_ch++) {
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
                                (depth) % POOL_STRIDE == 0 &&
                                (height) % POOL_STRIDE == 0 &&
                                (width) % POOL_STRIDE == 0) {

                                float max_vals[T_IN_CHANNELS];
                                #pragma HLS bind_storage variable=max_vals type=ram_2p impl=bram

                                for (int ch = 0; ch < T_IN_CHANNELS; ch++) {
                                    #pragma HLS pipeline II=1
                                    max_vals[ch] = -1e9f;  // FIXME: -FLT_MAX?

                                    for (int kd = 0; kd < POOL_KERNEL; kd++) {
                                        for (int kh = 0; kh < POOL_KERNEL; kh++) {
                                            for (int kw = 0; kw < POOL_KERNEL; kw++) {
                                                float window_val = window_buffer[batch][ch][kd][kh][kw];
                                                float max_val_temp = max_vals[ch];
                                                max_vals[ch] = fmax(max_val_temp, window_val);
                                            }
                                        }
                                    }

                                    int out_depth = (depth - POOL_KERNEL) / POOL_STRIDE + 1;
                                    int out_height = (height - POOL_KERNEL) / POOL_STRIDE + 1;
                                    int out_width = (width - POOL_KERNEL) / POOL_STRIDE + 1;

                                    float output_value = max_vals[ch];
                                    output[batch][ch][out_depth][out_height][out_width] = output_value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<int T_IN_CHANNELS>
void Upsample3D(float input[BATCH_SIZE][T_IN_CHANNELS][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
                float output[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]) {
    #pragma HLS array_partition variable=input cyclic factor=T_IN_CHANNELS dim=2
    #pragma HLS array_partition variable=output cyclic factor=T_IN_CHANNELS dim=2

    // Line buffer for streaming data access pattern
    float line_buffer[T_IN_CHANNELS][POOL_OUTPUT_WIDTH];
    #pragma HLS array_partition variable=line_buffer cyclic factor=T_IN_CHANNELS dim=1
    #pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

    // Main processing loops
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int src_d = 0; src_d < POOL_OUTPUT_DEPTH; src_d++) {
            for (int src_h = 0; src_h < POOL_OUTPUT_HEIGHT; src_h++) {
                // Load input line into buffer for reuse
                for (int src_w = 0; src_w < POOL_OUTPUT_WIDTH; src_w++) {
                    #pragma HLS pipeline II=1
                    for (int ch = 0; ch < T_IN_CHANNELS; ch++) {
                        #pragma HLS unroll
                        line_buffer[ch][src_w] = input[batch][ch][src_d][src_h][src_w];
                    }
                }

                // Generate upsampled outputs (2x2 expansion per source pixel)
                for (int h_offset = 0; h_offset < 2; h_offset++) {
                    int out_h = src_h * 2 + h_offset;

                    if (out_h < INPUT_HEIGHT) {
                        for (int src_w = 0; src_w < POOL_OUTPUT_WIDTH; src_w++) {
                            #pragma HLS pipeline II=1

                            int out_w_base = src_w * 2;

                            // Process all channels in parallel
                            for (int ch = 0; ch < T_IN_CHANNELS; ch++) {
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

template<int T_CHANNELS1, int T_CHANNELS2, int T_CONCAT_CHANNELS>
void ConcatenateTensors(float tensor1[BATCH_SIZE][T_CHANNELS1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                        float tensor2[BATCH_SIZE][T_CHANNELS2][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                        float output[BATCH_SIZE][T_CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]) {
    #pragma HLS array_partition variable=tensor1 cyclic factor=T_CHANNELS1 dim=2
    #pragma HLS array_partition variable=tensor2 cyclic factor=T_CHANNELS2 dim=2
    #pragma HLS array_partition variable=output cyclic factor=T_CONCAT_CHANNELS dim=2

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    #pragma HLS pipeline II=1
                    // Process tensor1 channels in parallel
                    for (int ch = 0; ch < T_CHANNELS1; ch++) {
                        #pragma HLS unroll
                        output[batch][ch][depth][height][width] =
                            tensor1[batch][ch][depth][height][width];
                    }

                    // Process tensor2 channels in parallel
                    for (int ch = 0; ch < T_CHANNELS2; ch++) {
                        #pragma HLS unroll
                        output[batch][T_CHANNELS1 + ch][depth][height][width] =
                            tensor2[batch][ch][depth][height][width];
                    }
                }
            }
        }
    }
}

template<int T_IN_CHANNELS, int T_OUT_CHANNELS>
void FinalConv1x1(float kernel[T_OUT_CHANNELS][T_IN_CHANNELS][1][1][1],
                  float bias[T_OUT_CHANNELS],
                  float input[BATCH_SIZE][T_IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                  float output[BATCH_SIZE][T_OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS array_partition variable=kernel cyclic factor=T_IN_CHANNELS dim=2
    #pragma HLS array_partition variable=bias complete dim=1

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    #pragma HLS pipeline II=1

                    float accum[T_OUT_CHANNELS];
                    #pragma HLS array_partition variable=accum complete dim=1
                    #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                    for (int out_ch = 0; out_ch < T_OUT_CHANNELS; out_ch++) {
                        accum[out_ch] = bias[out_ch];

                        for (int in_ch = 0; in_ch < T_IN_CHANNELS; in_ch++) {
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

void Sigmoid3D(float input[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
               float output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]) {
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
    #pragma HLS array_partition variable=sigmoid_lut complete dim=OUT_CHANNELS
    #pragma HLS bind_storage variable=sigmoid_lut type=ram_2p impl=lutram

    // Un-flattened nested loops - let Vitis HLS optimize automatically
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int ch = 0; ch < OUT_CHANNELS; ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
                        #pragma HLS pipeline II=1

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
            }
        }
    }
}

// Template function instantiations for DoubleConv3D
template void GroupNorm3D<IN_CHANNELS>(float[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[IN_CHANNELS], float[IN_CHANNELS], float[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
template void GroupNorm3D<F_MAP_h>(float[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[F_MAP_h], float[F_MAP_h], float[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
template void GroupNorm3D<F_MAP_0>(float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[F_MAP_0], float[F_MAP_0], float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
template void GroupNorm3D<F_MAP_0, POOL_OUTPUT_DEPTH, POOL_OUTPUT_HEIGHT, POOL_OUTPUT_WIDTH>(float[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH], float[F_MAP_0], float[F_MAP_0], float[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]);
template void GroupNorm3D<F_MAP_1, POOL_OUTPUT_DEPTH, POOL_OUTPUT_HEIGHT, POOL_OUTPUT_WIDTH>(float[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH], float[F_MAP_1], float[F_MAP_1], float[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]);
template void GroupNorm3D<CONCAT_CHANNELS>(float[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[CONCAT_CHANNELS], float[CONCAT_CHANNELS], float[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

template void Conv3D<IN_CHANNELS, F_MAP_h>(float[F_MAP_h][IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
template void Conv3D<F_MAP_h, F_MAP_0>(float[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
template void Conv3D<F_MAP_0, F_MAP_0, POOL_OUTPUT_DEPTH, POOL_OUTPUT_HEIGHT, POOL_OUTPUT_WIDTH>(float[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH], float[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]);
template void Conv3D<F_MAP_0, F_MAP_1, POOL_OUTPUT_DEPTH, POOL_OUTPUT_HEIGHT, POOL_OUTPUT_WIDTH>(float[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH], float[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]);
template void Conv3D<CONCAT_CHANNELS, F_MAP_0>(float[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
template void Conv3D<F_MAP_0, F_MAP_0>(float[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

// Template function instantiations for top function
// Input Path
template void DoubleConv3D2Head<IN_CHANNELS, F_MAP_h, F_MAP_0>(float[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[F_MAP_h][IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[IN_CHANNELS], float[IN_CHANNELS], float[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[F_MAP_h], float[F_MAP_h], float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
// Encoder Path
template void MaxPool3D<F_MAP_0>(float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]);
template void DoubleConv3D<F_MAP_0, F_MAP_0, F_MAP_1, POOL_OUTPUT_DEPTH, POOL_OUTPUT_HEIGHT, POOL_OUTPUT_WIDTH>(float[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH], float[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[F_MAP_0], float[F_MAP_0], float[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[F_MAP_0], float[F_MAP_0], float[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]);
// Decoder Path
template void Upsample3D<F_MAP_1>(float[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH], float[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
template void ConcatenateTensors<F_MAP_0, F_MAP_1, CONCAT_CHANNELS>(float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
template void DoubleConv3D<CONCAT_CHANNELS, F_MAP_0, F_MAP_0>(float[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[CONCAT_CHANNELS], float[CONCAT_CHANNELS], float[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[F_MAP_0], float[F_MAP_0], float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
// Output Path
template void DoubleConv3D<F_MAP_0, F_MAP_0, F_MAP_0>(float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[F_MAP_0], float[F_MAP_0], float[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL], float[F_MAP_0], float[F_MAP_0], float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
template void FinalConv1x1<F_MAP_0, OUT_CHANNELS>(float[OUT_CHANNELS][F_MAP_0][1][1][1], float[OUT_CHANNELS], float[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH], float[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);