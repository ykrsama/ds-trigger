#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include "unet3d_reduced.h"

using namespace std;

void initialize_test_data(
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float expected_output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {

    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < INPUT_CHANNELS; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        float val = 0.1f * (d + h + w);
                        input[b][c][d][h][w] = val;
                    }
                }
            }
        }
    }

    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < OUTPUT_CHANNELS; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        expected_output[b][c][d][h][w] = 0.5f + 0.1f * c;
                    }
                }
            }
        }
    }
}

void initialize_weights(
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
    float final_conv_bias[OUTPUT_CHANNELS]
) {

    float weight_scale = 1.0f / sqrt(KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE);

    for (int oc = 0; oc < F_MAPS_0; oc++) {
        for (int ic = 0; ic < INPUT_CHANNELS; ic++) {
            for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        input_conv1_weight[oc][ic][kd][kh][kw] = weight_scale * (0.1f + 0.01f * (oc + ic));
                    }
                }
            }
        }
        input_conv1_gamma[oc] = 1.0f;
        input_conv1_beta[oc] = 0.0f;
    }

    for (int oc = 0; oc < F_MAPS_0; oc++) {
        for (int ic = 0; ic < F_MAPS_0; ic++) {
            for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        input_conv2_weight[oc][ic][kd][kh][kw] = weight_scale * (0.1f + 0.01f * (oc + ic));
                    }
                }
            }
        }
        input_conv2_gamma[oc] = 1.0f;
        input_conv2_beta[oc] = 0.0f;
    }

    for (int oc = 0; oc < F_MAPS_1; oc++) {
        for (int ic = 0; ic < F_MAPS_0; ic++) {
            for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        encoder_conv1_weight[oc][ic][kd][kh][kw] = weight_scale * (0.1f + 0.01f * (oc + ic));
                    }
                }
            }
        }
        encoder_conv1_gamma[oc] = 1.0f;
        encoder_conv1_beta[oc] = 0.0f;
    }

    for (int oc = 0; oc < F_MAPS_1; oc++) {
        for (int ic = 0; ic < F_MAPS_1; ic++) {
            for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        encoder_conv2_weight[oc][ic][kd][kh][kw] = weight_scale * (0.1f + 0.01f * (oc + ic));
                    }
                }
            }
        }
        encoder_conv2_gamma[oc] = 1.0f;
        encoder_conv2_beta[oc] = 0.0f;
    }

    for (int oc = 0; oc < F_MAPS_0; oc++) {
        for (int ic = 0; ic < F_MAPS_0+F_MAPS_1; ic++) {
            for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        decoder_conv1_weight[oc][ic][kd][kh][kw] = weight_scale * (0.1f + 0.01f * (oc + ic));
                    }
                }
            }
        }
        decoder_conv1_gamma[oc] = 1.0f;
        decoder_conv1_beta[oc] = 0.0f;
    }

    for (int oc = 0; oc < F_MAPS_0; oc++) {
        for (int ic = 0; ic < F_MAPS_0; ic++) {
            for (int kd = 0; kd < KERNEL_SIZE; kd++) {
                for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        decoder_conv2_weight[oc][ic][kd][kh][kw] = weight_scale * (0.1f + 0.01f * (oc + ic));
                        output_conv1_weight[oc][ic][kd][kh][kw] = weight_scale * (0.1f + 0.01f * (oc + ic));
                        output_conv2_weight[oc][ic][kd][kh][kw] = weight_scale * (0.1f + 0.01f * (oc + ic));
                    }
                }
            }
        }
        decoder_conv2_gamma[oc] = 1.0f;
        decoder_conv2_beta[oc] = 0.0f;
        output_conv1_gamma[oc] = 1.0f;
        output_conv1_beta[oc] = 0.0f;
        output_conv2_gamma[oc] = 1.0f;
        output_conv2_beta[oc] = 0.0f;
    }

    for (int oc = 0; oc < OUTPUT_CHANNELS; oc++) {
        for (int ic = 0; ic < F_MAPS_0; ic++) {
            final_conv_weight[oc][ic][0][0][0] = weight_scale * (0.1f + 0.01f * (oc + ic));
        }
        final_conv_bias[oc] = 0.1f * oc;
    }
}

int main() {
    cout << "3D U-Net with Reduced Contraction Path - Test Bench" << endl;
    cout << "Input size: [" << BATCH_SIZE << "][" << INPUT_CHANNELS << "]["
         << INPUT_DEPTH << "][" << INPUT_HEIGHT << "][" << INPUT_WIDTH << "]" << endl;
    cout << "Output size: [" << BATCH_SIZE << "][" << OUTPUT_CHANNELS << "]["
         << INPUT_DEPTH << "][" << INPUT_HEIGHT << "][" << INPUT_WIDTH << "]" << endl;
    cout << "Feature maps: [" << F_MAPS_0 << ", " << F_MAPS_1 << "]" << endl << endl;

    static float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    static float expected_output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    static float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];

    static float input_conv1_weight[F_MAPS_0][INPUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
    static float input_conv1_gamma[F_MAPS_0];
    static float input_conv1_beta[F_MAPS_0];
    static float input_conv2_weight[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
    static float input_conv2_gamma[F_MAPS_0];
    static float input_conv2_beta[F_MAPS_0];

    static float encoder_conv1_weight[F_MAPS_1][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
    static float encoder_conv1_gamma[F_MAPS_1];
    static float encoder_conv1_beta[F_MAPS_1];
    static float encoder_conv2_weight[F_MAPS_1][F_MAPS_1][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
    static float encoder_conv2_gamma[F_MAPS_1];
    static float encoder_conv2_beta[F_MAPS_1];

    static float decoder_conv1_weight[F_MAPS_0][F_MAPS_0+F_MAPS_1][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
    static float decoder_conv1_gamma[F_MAPS_0];
    static float decoder_conv1_beta[F_MAPS_0];
    static float decoder_conv2_weight[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
    static float decoder_conv2_gamma[F_MAPS_0];
    static float decoder_conv2_beta[F_MAPS_0];

    static float output_conv1_weight[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
    static float output_conv1_gamma[F_MAPS_0];
    static float output_conv1_beta[F_MAPS_0];
    static float output_conv2_weight[F_MAPS_0][F_MAPS_0][KERNEL_SIZE][KERNEL_SIZE][KERNEL_SIZE];
    static float output_conv2_gamma[F_MAPS_0];
    static float output_conv2_beta[F_MAPS_0];

    static float final_conv_weight[OUTPUT_CHANNELS][F_MAPS_0][1][1][1];
    static float final_conv_bias[OUTPUT_CHANNELS];

    cout << "Initializing test data..." << endl;
    initialize_test_data(input, expected_output);

    cout << "Initializing weights..." << endl;
    initialize_weights(
        input_conv1_weight, input_conv1_gamma, input_conv1_beta,
        input_conv2_weight, input_conv2_gamma, input_conv2_beta,
        encoder_conv1_weight, encoder_conv1_gamma, encoder_conv1_beta,
        encoder_conv2_weight, encoder_conv2_gamma, encoder_conv2_beta,
        decoder_conv1_weight, decoder_conv1_gamma, decoder_conv1_beta,
        decoder_conv2_weight, decoder_conv2_gamma, decoder_conv2_beta,
        output_conv1_weight, output_conv1_gamma, output_conv1_beta,
        output_conv2_weight, output_conv2_gamma, output_conv2_beta,
        final_conv_weight, final_conv_bias
    );

    cout << "Running 3D U-Net inference..." << endl;

    unet3d_reduced(
        input,
        input_conv1_weight, input_conv1_gamma, input_conv1_beta,
        input_conv2_weight, input_conv2_gamma, input_conv2_beta,
        encoder_conv1_weight, encoder_conv1_gamma, encoder_conv1_beta,
        encoder_conv2_weight, encoder_conv2_gamma, encoder_conv2_beta,
        decoder_conv1_weight, decoder_conv1_gamma, decoder_conv1_beta,
        decoder_conv2_weight, decoder_conv2_gamma, decoder_conv2_beta,
        output_conv1_weight, output_conv1_gamma, output_conv1_beta,
        output_conv2_weight, output_conv2_gamma, output_conv2_beta,
        final_conv_weight, final_conv_bias,
        output
    );

    cout << "Inference completed!" << endl;

    cout << "Sample output values:" << endl;
    for (int c = 0; c < min(OUTPUT_CHANNELS, 3); c++) {
        cout << "Channel " << c << ": ";
        for (int i = 0; i < min(5, INPUT_WIDTH); i++) {
            cout << output[0][c][0][0][i] << " ";
        }
        cout << "..." << endl;
    }

    float total_sum = 0.0f;
    int total_elements = BATCH_SIZE * OUTPUT_CHANNELS * INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH;

    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < OUTPUT_CHANNELS; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        total_sum += fabs(output[b][c][d][h][w]);
                    }
                }
            }
        }
    }

    float avg_magnitude = total_sum / total_elements;
    cout << "Average output magnitude: " << avg_magnitude << endl;

    cout << "Test completed successfully!" << endl;

    return 0;
}