#ifndef OUTPUT_PATH_H
#define OUTPUT_PATH_H

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "model_parameter.h"

// Output path function declarations
void OutputConv3D_1(
    float kernel[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void OutputConv3D_2(
    float kernel[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void FinalConv1x1(
    float kernel[OUTPUT_CHANNELS][F_MAP_0][1][1][1],
    float bias[OUTPUT_CHANNELS],
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void OutputGroupNorm3D_1(
    float input_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[F_MAP_0],
    float beta[F_MAP_0],
    float output_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void OutputGroupNorm3D_2(
    float input_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[F_MAP_0],
    float beta[F_MAP_0],
    float output_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void OutputDoubleConv3D(
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma1[F_MAP_0],
    float beta1[F_MAP_0],
    float kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float gamma2[F_MAP_0],
    float beta2[F_MAP_0],
    float kernel2[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

void Sigmoid3D(
    float input[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
);

// Output path function implementations

void OutputConv3D_1(
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

void OutputConv3D_2(
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

void OutputGroupNorm3D_1(
    float input_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[F_MAP_0],
    float beta[F_MAP_0],
    float output_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS dataflow

    // Calculate group parameters
    int CHANNELS_PER_GROUP = F_MAP_0 / NUM_GROUPS;
    float N = (float)(INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * CHANNELS_PER_GROUP);

    // Define on-chip buffers
    float gn_buffer[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=gn_buffer type=ram_2p impl=uram

    // Statistics arrays
    float group_sum[NUM_GROUPS];
    float group_sq_sum[NUM_GROUPS];
    #pragma HLS array_partition variable=group_sum complete
    #pragma HLS array_partition variable=group_sq_sum complete

    for (int g = 0; g < NUM_GROUPS; g++) {
        #pragma HLS unroll
        group_sum[g] = 0.0f;
        group_sq_sum[g] = 0.0f;
    }

    // Compute statistics and buffer data
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

    // Compute mean and inverse std
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

    // Apply normalization and affine transform
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

void OutputGroupNorm3D_2(
    float input_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[F_MAP_0],
    float beta[F_MAP_0],
    float output_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS dataflow

    // Calculate group parameters
    int CHANNELS_PER_GROUP = F_MAP_0 / NUM_GROUPS;
    float N = (float)(INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * CHANNELS_PER_GROUP);

    // Define on-chip buffers
    float gn_buffer[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=gn_buffer type=ram_2p impl=uram

    // Statistics arrays
    float group_sum[NUM_GROUPS];
    float group_sq_sum[NUM_GROUPS];
    #pragma HLS array_partition variable=group_sum complete
    #pragma HLS array_partition variable=group_sq_sum complete

    for (int g = 0; g < NUM_GROUPS; g++) {
        #pragma HLS unroll
        group_sum[g] = 0.0f;
        group_sq_sum[g] = 0.0f;
    }

    // Compute statistics and buffer data
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

    // Compute mean and inverse std
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

    // Apply normalization and affine transform
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

void OutputDoubleConv3D(
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma1[F_MAP_0],
    float beta1[F_MAP_0],
    float kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float gamma2[F_MAP_0],
    float beta2[F_MAP_0],
    float kernel2[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS dataflow

    // Intermediate buffers for pipeline
    float gn1_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=gn1_out depth=10 type=fifo

    float conv1_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=conv1_out depth=10 type=fifo

    float gn2_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=gn2_out depth=10 type=fifo

    // Pipeline: GroupNorm1 + Conv1 + GroupNorm2 + Conv2 (gcr order)
    OutputGroupNorm3D_1(input, gamma1, beta1, gn1_out);
    OutputConv3D_1(kernel1, gn1_out, conv1_out);
    OutputGroupNorm3D_2(conv1_out, gamma2, beta2, gn2_out);
    OutputConv3D_2(kernel2, gn2_out, output);
}

#endif // OUTPUT_PATH_H