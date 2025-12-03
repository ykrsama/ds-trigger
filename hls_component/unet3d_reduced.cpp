#include "include/unet3d_reduced.h"


// GroupNorm3D template
template<int T_INPUT_CHANNELS>
void GroupNorm3D(float input_data[BATCH_SIZE][T_INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
                 float gamma[T_INPUT_CHANNELS],
                 float beta[T_INPUT_CHANNELS],
                 float output_data[BATCH_SIZE][T_INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]) {
    // calculate channel number
    int CHANNELS_PER_GROUP = T_INPUT_CHANNELS / NUM_GROUPS;
    float N = (float) (INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * CHANNELS_PER_GROUP);

    // 1. On chip buffer
    float gn_buffer[BATCH_SIZE][T_INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
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
        for (int ch = 0; ch < T_INPUT_CHANNELS; ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
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
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    #pragma HLS pipeline II=1
                    for (int ch = 0; ch < INPUT_CHANNELS; ch++) {
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
template<int T_INPUT_CHANNELS, int T_OUT_CHANNELS>
void Conv3d(float kernel[T_OUT_CHANNELS][T_INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
            float input[BATCH_SIZE][T_INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
            float output[BATCH_SIZE][T_OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]) {
    #pragma HLS array_partition variable=kernel cyclic factor=T_INPUT_CHANNELS dim=2
    #pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=3
    #pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=4
    #pragma HLS array_partition variable=kernel cyclic factor=CONV_KERNEL dim=5

    // fill input
    float padded_input[BATCH_SIZE][T_INPUT_CHANNELS][PADDED_DEPTH][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS stream variable=padded_input depth=10 type=fifo
    #pragma HLS bind_storage variable=padded_input type=ram_t2p impl=bram

    // Filling operation
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < PADDED_DEPTH; depth++) {
            for (int height = 0; height < PADDED_HEIGHT; height++) {
                for (int width = 0; width < PADDED_WIDTH; width++) {
                    for (int in_ch = 0; in_ch < T_INPUT_CHANNELS; in_ch++) {
                        float pad_value = (float) 0.000000;
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
    float cube_buffer[BATCH_SIZE][T_INPUT_CHANNELS][CONV_KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
    #pragma HLS bind_storage variable=cube_buffer type=ram_2p impl=lutram

    float line_buffer[BATCH_SIZE][T_INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][PADDED_WIDTH];
    #pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

    float window_buffer[BATCH_SIZE][T_INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    #pragma HLS array_partition variable=window_buffer cyclic factor=T_INPUT_CHANNELS dim=2
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
                    for (int in_ch = 0; in_ch < T_INPUT_CHANNELS; in_ch++) {
                        for (int kd = 0; kd < CONV_KERNEL - 1; kd++) {
                            cube_buffer[batch][in_ch][kd][height][width] =
                                cube_buffer[batch][in_ch][kd + 1][height][width];
                        }
                        cube_buffer[batch][in_ch][CONV_KERNEL - 1][height][width] =
                            padded_input[batch][in_ch][depth][height][width];
                    }

                    if (depth >= CONV_KERNEL - 1) {
                        // Update line buffer
                        for (int in_ch = 0; in_ch < T_INPUT_CHANNELS; in_ch++) {
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
                            for (int in_ch = 0; in_ch < T_INPUT_CHANNELS; in_ch++) {
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
                                (depth) % CONV_STRIDE == 0 &&
                                (height) % CONV_STRIDE == 0 &&
                                (width) % CONV_STRIDE == 0) {

                                float accum[T_OUT_CHANNELS];
                                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram

                                for (int out_ch = 0; out_ch < T_OUT_CHANNELS; out_ch++) {
                                    #pragma HLS pipeline II=1
                                    accum[out_ch] = (float)0.000000;

                                    for (int in_ch = 0; in_ch < T_INPUT_CHANNELS; in_ch++) {
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

                                    int out_depth = (depth - CONV_KERNEL) / STRIDE + 1;
                                    int out_height = (height - CONV_KERNEL) / STRIDE + 1;
                                    int out_width = (width - CONV_KERNEL) / STRIDE + 1;

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

    // GroupNorm parameters
    float input_conv1_gamma[INPUT_CHANNELS],
    float input_conv1_beta[INPUT_CHANNELS],
    float input_conv2_gamma[F_MAP_h],
    float input_conv2_beta[F_MAP_h],

    float encoder_conv1_gamma[F_MAP_0],
    float encoder_conv1_beta[F_MAP_0],
    float encoder_conv2_gamma[F_MAP_0],
    float encoder_conv2_beta[F_MAP_0],

    float decoder_conv1_gamma[CONCAT_CHANNELS],
    float decoder_conv1_beta[CONCAT_CHANNELS],
    float decoder_conv2_gamma[F_MAP_0],
    float decoder_conv2_beta[F_MAP_0],

    float output_conv1_gamma[F_MAP_0],
    float output_conv1_beta[F_MAP_0],
    float output_conv2_gamma[F_MAP_0],
    float output_conv2_beta[F_MAP_0],

    // Output
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return

    #pragma HLS interface bram port=input
    #pragma HLS bind_storage variable=input type=ram_t2p impl=bram

    #pragma HLS interface bram port=input_conv1_weight
    #pragma HLS bind_storage variable=input_conv1_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=input_conv2_weight
    #pragma HLS bind_storage variable=input_conv2_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=encoder_conv1_weight
    #pragma HLS bind_storage variable=encoder_conv1_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=encoder_conv2_weight
    #pragma HLS bind_storage variable=encoder_conv2_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=decoder_conv1_weight
    #pragma HLS bind_storage variable=decoder_conv1_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=decoder_conv2_weight
    #pragma HLS bind_storage variable=decoder_conv2_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=output_conv1_weight
    #pragma HLS bind_storage variable=output_conv1_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=output_conv2_weight
    #pragma HLS bind_storage variable=output_conv2_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=final_conv_weight
    #pragma HLS bind_storage variable=final_conv_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=final_conv_bias
    #pragma HLS bind_storage variable=final_conv_bias type=ram_t2p impl=bram

    #pragma HLS interface bram port=input_conv1_gamma
    #pragma HLS bind_storage variable=input_conv1_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=input_conv1_beta
    #pragma HLS bind_storage variable=input_conv1_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=input_conv2_gamma
    #pragma HLS bind_storage variable=input_conv2_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=input_conv2_beta
    #pragma HLS bind_storage variable=input_conv2_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=encoder_conv1_gamma
    #pragma HLS bind_storage variable=encoder_conv1_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=encoder_conv1_beta
    #pragma HLS bind_storage variable=encoder_conv1_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=encoder_conv2_gamma
    #pragma HLS bind_storage variable=encoder_conv2_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=encoder_conv2_beta
    #pragma HLS bind_storage variable=encoder_conv2_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=decoder_conv1_gamma
    #pragma HLS bind_storage variable=decoder_conv1_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=decoder_conv1_beta
    #pragma HLS bind_storage variable=decoder_conv1_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=decoder_conv2_gamma
    #pragma HLS bind_storage variable=decoder_conv2_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=decoder_conv2_beta
    #pragma HLS bind_storage variable=decoder_conv2_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=output_conv1_gamma
    #pragma HLS bind_storage variable=output_conv1_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=output_conv1_beta
    #pragma HLS bind_storage variable=output_conv1_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=output_conv2_gamma
    #pragma HLS bind_storage variable=output_conv2_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=output_conv2_beta
    #pragma HLS bind_storage variable=output_conv2_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=output
    #pragma HLS bind_storage variable=output type=ram_t2p impl=bram

    #pragma HLS dataflow

    // Input path buffers - need two streams for skip connection
    float input_conv_out_main[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=input_conv_out_main depth=10 type=fifo
    #pragma HLS bind_storage variable=input_conv_out_main type=ram_2p impl=bram

    float input_conv_out_skip[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=input_conv_out_skip depth=10 type=fifo
    #pragma HLS bind_storage variable=input_conv_out_skip type=ram_2p impl=bram

    // Encoder path buffers
    float encoder_pool_out[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_pool_out depth=10 type=fifo
    #pragma HLS bind_storage variable=encoder_pool_out type=ram_2p impl=bram

    float encoder_conv_out[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=encoder_conv_out type=ram_2p impl=bram

    // Decoder path buffers
    float decoder_upsample_out[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_upsample_out depth=10 type=fifo
    #pragma HLS bind_storage variable=decoder_upsample_out type=ram_2p impl=bram

    float concat_out[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=concat_out depth=10 type=fifo
    #pragma HLS bind_storage variable=concat_out type=ram_2p impl=bram

    float decoder_conv_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=decoder_conv_out type=ram_2p impl=bram

    // Output path buffers
    float output_conv_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=output_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=output_conv_out type=ram_2p impl=bram

    float final_conv_out[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=final_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=final_conv_out type=ram_2p impl=bram

    // UNet forward pass
    // Input Path
    // [INPUT_CHANNELS] -> DoubleConv -> [F_MAP_0] (split into two streams)
    InputDoubleConv3D(input, input_conv1_weight, input_conv1_gamma, input_conv1_beta,
                      input_conv2_weight, input_conv2_gamma, input_conv2_beta,
                      input_conv_out_main, input_conv_out_skip);

    // [F_MAP_0] -> MaxPool -> [F_MAP_0] -> DoubleConv -> [F_MAP_1]
    EncoderMaxPool3D(input_conv_out_main, encoder_pool_out);
    EncoderDoubleConv3D(encoder_conv1_weight, encoder_conv1_gamma, encoder_conv1_beta,
                       encoder_conv2_weight, encoder_conv2_gamma, encoder_conv2_beta,
                       encoder_pool_out, encoder_conv_out);

    // [F_MAP_1] -> Upsampling -> [F_MAP_1] -> Concat -> [CONCAT_CHANNELS]
    DecoderUpsample3D(encoder_conv_out, decoder_upsample_out);
    ConcatenateTensors(input_conv_out_skip, decoder_upsample_out, concat_out);

    DecoderDoubleConv3D(decoder_conv1_weight, decoder_conv1_gamma, decoder_conv1_beta,
                        decoder_conv2_weight, decoder_conv2_gamma, decoder_conv2_beta,
                        concat_out, decoder_conv_out);

    OutputDoubleConv3D(decoder_conv_out, output_conv1_gamma, output_conv1_beta, output_conv1_weight,
                       output_conv2_gamma, output_conv2_beta, output_conv2_weight, output_conv_out);
    FinalConv1x1(final_conv_weight, final_conv_bias, output_conv_out, final_conv_out);
    Sigmoid3D(final_conv_out, output);
}