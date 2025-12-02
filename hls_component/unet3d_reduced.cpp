#include "unet3d_reduced.h"

// Input convolution block: INPUT_CHANNELS -> F_MAP_0 (1 -> 64)
// Implements two 3x3 convolutions with ReLU activation
void InputConv3D(
    float kernel1[F_MAP_0][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float kernel2[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS dataflow

    // Intermediate buffer for first convolution output
    float conv1_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=bram

    // First convolution: INPUT_CHANNELS -> F_MAP_0
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
                        #pragma HLS pipeline II=1

                        float accum = 0.0f;

                        // Convolution computation with padding
                        for (int in_ch = 0; in_ch < INPUT_CHANNELS; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                    for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                        int in_depth = depth + kd - CONV_PADDING;
                                        int in_height = height + kh - CONV_PADDING;
                                        int in_width = width + kw - CONV_PADDING;

                                        float input_val = 0.0f;
                                        if (in_depth >= 0 && in_depth < INPUT_DEPTH &&
                                            in_height >= 0 && in_height < INPUT_HEIGHT &&
                                            in_width >= 0 && in_width < INPUT_WIDTH) {
                                            input_val = input[batch][in_ch][in_depth][in_height][in_width];
                                        }

                                        accum += input_val * kernel1[out_ch][in_ch][kd][kh][kw];
                                    }
                                }
                            }
                        }

                        // ReLU activation
                        conv1_out[batch][out_ch][depth][height][width] = (accum > 0.0f) ? accum : 0.0f;
                    }
                }
            }
        }
    }

    // Second convolution: F_MAP_0 -> F_MAP_0
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
                        #pragma HLS pipeline II=1

                        float accum = 0.0f;

                        // Convolution computation with padding
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                    for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                        int in_depth = depth + kd - CONV_PADDING;
                                        int in_height = height + kh - CONV_PADDING;
                                        int in_width = width + kw - CONV_PADDING;

                                        float input_val = 0.0f;
                                        if (in_depth >= 0 && in_depth < INPUT_DEPTH &&
                                            in_height >= 0 && in_height < INPUT_HEIGHT &&
                                            in_width >= 0 && in_width < INPUT_WIDTH) {
                                            input_val = conv1_out[batch][in_ch][in_depth][in_height][in_width];
                                        }

                                        accum += input_val * kernel2[out_ch][in_ch][kd][kh][kw];
                                    }
                                }
                            }
                        }

                        // ReLU activation
                        output[batch][out_ch][depth][height][width] = (accum > 0.0f) ? accum : 0.0f;
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
    #pragma HLS dataflow

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int ch = 0; ch < F_MAP_0; ch++) {
            for (int out_d = 0; out_d < POOL_OUTPUT_DEPTH; out_d++) {
                for (int out_h = 0; out_h < POOL_OUTPUT_HEIGHT; out_h++) {
                    for (int out_w = 0; out_w < POOL_OUTPUT_WIDTH; out_w++) {
                        #pragma HLS pipeline II=1

                        float max_val = -1e9f;

                        for (int kd = 0; kd < POOL_KERNEL; kd++) {
                            for (int kh = 0; kh < POOL_KERNEL; kh++) {
                                for (int kw = 0; kw < POOL_KERNEL; kw++) {
                                    int in_depth = out_d * POOL_STRIDE + kd;
                                    int in_height = out_h * POOL_STRIDE + kh;
                                    int in_width = out_w * POOL_STRIDE + kw;

                                    if (in_depth < INPUT_DEPTH &&
                                        in_height < INPUT_HEIGHT &&
                                        in_width < INPUT_WIDTH) {
                                        float val = input[batch][ch][in_depth][in_height][in_width];
                                        if (val > max_val) {
                                            max_val = val;
                                        }
                                    }
                                }
                            }
                        }

                        output[batch][ch][out_d][out_h][out_w] = max_val;
                    }
                }
            }
        }
    }
}

// Encoder convolution block: F_MAP_0 -> F_MAP_1 (64 -> 128)
void EncoderConv3D(
    float kernel1[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float kernel2[F_MAP_1][F_MAP_1][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]
) {
    #pragma HLS dataflow

    // Intermediate buffer for first convolution output
    float conv1_out[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=bram

    // First convolution: F_MAP_0 -> F_MAP_1
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int out_ch = 0; out_ch < F_MAP_1; out_ch++) {
            for (int depth = 0; depth < POOL_OUTPUT_DEPTH; depth++) {
                for (int height = 0; height < POOL_OUTPUT_HEIGHT; height++) {
                    for (int width = 0; width < POOL_OUTPUT_WIDTH; width++) {
                        #pragma HLS pipeline II=1

                        float accum = 0.0f;

                        // Convolution computation with padding
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                    for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                        int in_depth = depth + kd - CONV_PADDING;
                                        int in_height = height + kh - CONV_PADDING;
                                        int in_width = width + kw - CONV_PADDING;

                                        float input_val = 0.0f;
                                        if (in_depth >= 0 && in_depth < POOL_OUTPUT_DEPTH &&
                                            in_height >= 0 && in_height < POOL_OUTPUT_HEIGHT &&
                                            in_width >= 0 && in_width < POOL_OUTPUT_WIDTH) {
                                            input_val = input[batch][in_ch][in_depth][in_height][in_width];
                                        }

                                        accum += input_val * kernel1[out_ch][in_ch][kd][kh][kw];
                                    }
                                }
                            }
                        }

                        // ReLU activation
                        conv1_out[batch][out_ch][depth][height][width] = (accum > 0.0f) ? accum : 0.0f;
                    }
                }
            }
        }
    }

    // Second convolution: F_MAP_1 -> F_MAP_1
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int out_ch = 0; out_ch < F_MAP_1; out_ch++) {
            for (int depth = 0; depth < POOL_OUTPUT_DEPTH; depth++) {
                for (int height = 0; height < POOL_OUTPUT_HEIGHT; height++) {
                    for (int width = 0; width < POOL_OUTPUT_WIDTH; width++) {
                        #pragma HLS pipeline II=1

                        float accum = 0.0f;

                        // Convolution computation with padding
                        for (int in_ch = 0; in_ch < F_MAP_1; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                    for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                        int in_depth = depth + kd - CONV_PADDING;
                                        int in_height = height + kh - CONV_PADDING;
                                        int in_width = width + kw - CONV_PADDING;

                                        float input_val = 0.0f;
                                        if (in_depth >= 0 && in_depth < POOL_OUTPUT_DEPTH &&
                                            in_height >= 0 && in_height < POOL_OUTPUT_HEIGHT &&
                                            in_width >= 0 && in_width < POOL_OUTPUT_WIDTH) {
                                            input_val = conv1_out[batch][in_ch][in_depth][in_height][in_width];
                                        }

                                        accum += input_val * kernel2[out_ch][in_ch][kd][kh][kw];
                                    }
                                }
                            }
                        }

                        // ReLU activation
                        output[batch][out_ch][depth][height][width] = (accum > 0.0f) ? accum : 0.0f;
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
    #pragma HLS dataflow

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int ch = 0; ch < F_MAP_1; ch++) {
            for (int out_d = 0; out_d < INPUT_DEPTH; out_d++) {
                for (int out_h = 0; out_h < INPUT_HEIGHT; out_h++) {
                    for (int out_w = 0; out_w < INPUT_WIDTH; out_w++) {
                        #pragma HLS pipeline II=1

                        // Nearest neighbor interpolation
                        int src_d = out_d / 2;
                        int src_h = out_h / 2;
                        int src_w = out_w / 2;

                        // Clamp to valid range
                        if (src_d >= POOL_OUTPUT_DEPTH) src_d = POOL_OUTPUT_DEPTH - 1;
                        if (src_h >= POOL_OUTPUT_HEIGHT) src_h = POOL_OUTPUT_HEIGHT - 1;
                        if (src_w >= POOL_OUTPUT_WIDTH) src_w = POOL_OUTPUT_WIDTH - 1;

                        output[batch][ch][out_d][out_h][out_w] =
                            input[batch][ch][src_d][src_h][src_w];
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
    #pragma HLS dataflow

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    #pragma HLS pipeline II=1

                    // Copy first tensor (F_MAP_0 channels)
                    for (int ch = 0; ch < F_MAP_0; ch++) {
                        output[batch][ch][depth][height][width] =
                            tensor1[batch][ch][depth][height][width];
                    }

                    // Copy second tensor (F_MAP_1 channels)
                    for (int ch = 0; ch < F_MAP_1; ch++) {
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
    #pragma HLS dataflow

    // Intermediate buffer for first convolution output
    float conv1_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=bram

    // First convolution: CONCAT_CHANNELS -> F_MAP_0
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
                        #pragma HLS pipeline II=1

                        float accum = 0.0f;

                        // Convolution computation with padding
                        for (int in_ch = 0; in_ch < CONCAT_CHANNELS; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                    for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                        int in_depth = depth + kd - CONV_PADDING;
                                        int in_height = height + kh - CONV_PADDING;
                                        int in_width = width + kw - CONV_PADDING;

                                        float input_val = 0.0f;
                                        if (in_depth >= 0 && in_depth < INPUT_DEPTH &&
                                            in_height >= 0 && in_height < INPUT_HEIGHT &&
                                            in_width >= 0 && in_width < INPUT_WIDTH) {
                                            input_val = input[batch][in_ch][in_depth][in_height][in_width];
                                        }

                                        accum += input_val * kernel1[out_ch][in_ch][kd][kh][kw];
                                    }
                                }
                            }
                        }

                        // ReLU activation
                        conv1_out[batch][out_ch][depth][height][width] = (accum > 0.0f) ? accum : 0.0f;
                    }
                }
            }
        }
    }

    // Second convolution: F_MAP_0 -> F_MAP_0
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
                        #pragma HLS pipeline II=1

                        float accum = 0.0f;

                        // Convolution computation with padding
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                    for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                        int in_depth = depth + kd - CONV_PADDING;
                                        int in_height = height + kh - CONV_PADDING;
                                        int in_width = width + kw - CONV_PADDING;

                                        float input_val = 0.0f;
                                        if (in_depth >= 0 && in_depth < INPUT_DEPTH &&
                                            in_height >= 0 && in_height < INPUT_HEIGHT &&
                                            in_width >= 0 && in_width < INPUT_WIDTH) {
                                            input_val = conv1_out[batch][in_ch][in_depth][in_height][in_width];
                                        }

                                        accum += input_val * kernel2[out_ch][in_ch][kd][kh][kw];
                                    }
                                }
                            }
                        }

                        // ReLU activation
                        output[batch][out_ch][depth][height][width] = (accum > 0.0f) ? accum : 0.0f;
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
    #pragma HLS dataflow

    // Intermediate buffer for first convolution output
    float conv1_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=bram

    // First convolution: F_MAP_0 -> F_MAP_0
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
                        #pragma HLS pipeline II=1

                        float accum = 0.0f;

                        // Convolution computation with padding
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                    for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                        int in_depth = depth + kd - CONV_PADDING;
                                        int in_height = height + kh - CONV_PADDING;
                                        int in_width = width + kw - CONV_PADDING;

                                        float input_val = 0.0f;
                                        if (in_depth >= 0 && in_depth < INPUT_DEPTH &&
                                            in_height >= 0 && in_height < INPUT_HEIGHT &&
                                            in_width >= 0 && in_width < INPUT_WIDTH) {
                                            input_val = input[batch][in_ch][in_depth][in_height][in_width];
                                        }

                                        accum += input_val * kernel1[out_ch][in_ch][kd][kh][kw];
                                    }
                                }
                            }
                        }

                        // ReLU activation
                        conv1_out[batch][out_ch][depth][height][width] = (accum > 0.0f) ? accum : 0.0f;
                    }
                }
            }
        }
    }

    // Second convolution: F_MAP_0 -> F_MAP_0
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int out_ch = 0; out_ch < F_MAP_0; out_ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
                        #pragma HLS pipeline II=1

                        float accum = 0.0f;

                        // Convolution computation with padding
                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                                for (int kh = 0; kh < CONV_KERNEL; kh++) {
                                    for (int kw = 0; kw < CONV_KERNEL; kw++) {
                                        int in_depth = depth + kd - CONV_PADDING;
                                        int in_height = height + kh - CONV_PADDING;
                                        int in_width = width + kw - CONV_PADDING;

                                        float input_val = 0.0f;
                                        if (in_depth >= 0 && in_depth < INPUT_DEPTH &&
                                            in_height >= 0 && in_height < INPUT_HEIGHT &&
                                            in_width >= 0 && in_width < INPUT_WIDTH) {
                                            input_val = conv1_out[batch][in_ch][in_depth][in_height][in_width];
                                        }

                                        accum += input_val * kernel2[out_ch][in_ch][kd][kh][kw];
                                    }
                                }
                            }
                        }

                        // ReLU activation
                        output[batch][out_ch][depth][height][width] = (accum > 0.0f) ? accum : 0.0f;
                    }
                }
            }
        }
    }
}

// Final 1x1 convolution: F_MAP_0 -> OUTPUT_CHANNELS (64 -> 2)
void FinalConv1x1(
    float kernel[OUTPUT_CHANNELS][F_MAP_0][1][1][1],
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS dataflow

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int out_ch = 0; out_ch < OUTPUT_CHANNELS; out_ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
                        #pragma HLS pipeline II=1

                        float accum = 0.0f;

                        for (int in_ch = 0; in_ch < F_MAP_0; in_ch++) {
                            accum += input[batch][in_ch][depth][height][width] *
                                    kernel[out_ch][in_ch][0][0][0];
                        }

                        output[batch][out_ch][depth][height][width] = accum;
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
    #pragma HLS dataflow

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int ch = 0; ch < OUTPUT_CHANNELS; ch++) {
            for (int depth = 0; depth < INPUT_DEPTH; depth++) {
                for (int height = 0; height < INPUT_HEIGHT; height++) {
                    for (int width = 0; width < INPUT_WIDTH; width++) {
                        #pragma HLS pipeline II=1

                        float x = input[batch][ch][depth][height][width];
                        // Sigmoid approximation: 1 / (1 + exp(-x))
                        // For HLS, use a more efficient approximation
                        float sigmoid_val;
                        if (x > 5.0f) {
                            sigmoid_val = 1.0f;
                        } else if (x < -5.0f) {
                            sigmoid_val = 0.0f;
                        } else {
                            sigmoid_val = 1.0f / (1.0f + hls::exp(-x));
                        }

                        output[batch][ch][depth][height][width] = sigmoid_val;
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
    float input_conv1_weight[F_MAP_0][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float encoder_conv1_weight[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float encoder_conv2_weight[F_MAP_1][F_MAP_1][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float decoder_conv1_weight[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float decoder_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float output_conv1_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float output_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float final_conv_weight[OUTPUT_CHANNELS][F_MAP_0][1][1][1],

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
    FinalConv1x1(final_conv_weight, output_conv_out, final_conv_out);
    Sigmoid3D(final_conv_out, output);
}