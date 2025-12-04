#include "doubleconv_component.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    // Initialize test data
    float input[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    float kernel1[F_MAP_h][IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    float gamma1[IN_CHANNELS];
    float beta1[IN_CHANNELS];
    float kernel2[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    float gamma2[F_MAP_h];
    float beta2[F_MAP_h];
    float output1[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    float output2[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];

    // Seed random number generator
    srand(time(NULL));

    // Initialize input with random values
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < IN_CHANNELS; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        input[b][c][d][h][w] = (float)rand() / RAND_MAX - 0.5f; // Random values between -0.5 and 0.5
                    }
                }
            }
        }
    }

    // Initialize weights and parameters with random values
    for (int out = 0; out < F_MAP_h; out++) {
        for (int in = 0; in < IN_CHANNELS; in++) {
            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                for (int kh = 0; kh < CONV_KERNEL; kh++) {
                    for (int kw = 0; kw < CONV_KERNEL; kw++) {
                        kernel1[out][in][kd][kh][kw] = (float)rand() / RAND_MAX - 0.5f;
                    }
                }
            }
        }
    }

    for (int out = 0; out < F_MAP_0; out++) {
        for (int in = 0; in < F_MAP_h; in++) {
            for (int kd = 0; kd < CONV_KERNEL; kd++) {
                for (int kh = 0; kh < CONV_KERNEL; kh++) {
                    for (int kw = 0; kw < CONV_KERNEL; kw++) {
                        kernel2[out][in][kd][kh][kw] = (float)rand() / RAND_MAX - 0.5f;
                    }
                }
            }
        }
    }

    // Initialize gamma and beta parameters
    for (int c = 0; c < IN_CHANNELS; c++) {
        gamma1[c] = 1.0f;
        beta1[c] = 0.0f;
    }

    for (int c = 0; c < F_MAP_h; c++) {
        gamma2[c] = 1.0f;
        beta2[c] = 0.0f;
    }

    // Initialize outputs
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < F_MAP_0; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        output1[b][c][d][h][w] = 0.0f;
                        output2[b][c][d][h][w] = 0.0f;
                    }
                }
            }
        }
    }

    std::cout << "Starting DoubleConv Component test..." << std::endl;

    // Call the top function
    DoubleConvComponent(
        input,
        kernel1,
        gamma1,
        beta1,
        kernel2,
        gamma2,
        beta2,
        output1,
        output2
    );

    std::cout << "DoubleConv Component test completed successfully!" << std::endl;

    // Print some sample outputs for verification
    std::cout << "Sample output1[0][0][0][0][0]: " << output1[0][0][0][0][0] << std::endl;
    std::cout << "Sample output2[0][0][0][0][0]: " << output2[0][0][0][0][0] << std::endl;

    return 0;
}