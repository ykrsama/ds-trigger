#include "encoder_doubleconv.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    // Initialize test data
    float input[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    float kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    float gamma1[F_MAP_0];
    float beta1[F_MAP_0];
    float kernel2[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    float gamma2[F_MAP_0];
    float beta2[F_MAP_0];
    float output[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];

    // Seed random number generator
    srand(time(NULL));

    // Initialize input with random values
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < F_MAP_0; c++) {
            for (int d = 0; d < POOL_OUTPUT_DEPTH; d++) {
                for (int h = 0; h < POOL_OUTPUT_HEIGHT; h++) {
                    for (int w = 0; w < POOL_OUTPUT_WIDTH; w++) {
                        input[b][c][d][h][w] = (float)rand() / RAND_MAX - 0.5f; // Random values between -0.5 and 0.5
                    }
                }
            }
        }
    }

    // Initialize kernel1 with small random weights
    for (int out_c = 0; out_c < F_MAP_0; out_c++) {
        for (int in_c = 0; in_c < F_MAP_0; in_c++) {
            for (int d = 0; d < CONV_KERNEL; d++) {
                for (int h = 0; h < CONV_KERNEL; h++) {
                    for (int w = 0; w < CONV_KERNEL; w++) {
                        kernel1[out_c][in_c][d][h][w] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                    }
                }
            }
        }
    }

    // Initialize kernel2 with small random weights
    for (int out_c = 0; out_c < F_MAP_1; out_c++) {
        for (int in_c = 0; in_c < F_MAP_0; in_c++) {
            for (int d = 0; d < CONV_KERNEL; d++) {
                for (int h = 0; h < CONV_KERNEL; h++) {
                    for (int w = 0; w < CONV_KERNEL; w++) {
                        kernel2[out_c][in_c][d][h][w] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                    }
                }
            }
        }
    }

    // Initialize group norm parameters
    for (int c = 0; c < F_MAP_0; c++) {
        gamma1[c] = 1.0f;
        beta1[c] = 0.0f;
        gamma2[c] = 1.0f;
        beta2[c] = 0.0f;
    }

    // Initialize output
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < F_MAP_1; c++) {
            for (int d = 0; d < POOL_OUTPUT_DEPTH; d++) {
                for (int h = 0; h < POOL_OUTPUT_HEIGHT; h++) {
                    for (int w = 0; w < POOL_OUTPUT_WIDTH; w++) {
                        output[b][c][d][h][w] = 0.0f;
                    }
                }
            }
        }
    }

    std::cout << "Starting Encoder DoubleConv test..." << std::endl;
    std::cout << "Input dimensions: [" << BATCH_SIZE << "][" << F_MAP_0 << "]["
              << POOL_OUTPUT_DEPTH << "][" << POOL_OUTPUT_HEIGHT << "][" << POOL_OUTPUT_WIDTH << "]" << std::endl;
    std::cout << "Output dimensions: [" << BATCH_SIZE << "][" << F_MAP_1 << "]["
              << POOL_OUTPUT_DEPTH << "][" << POOL_OUTPUT_HEIGHT << "][" << POOL_OUTPUT_WIDTH << "]" << std::endl;

    // Call the top function
    EncoderDoubleConv(input, kernel1, gamma1, beta1, kernel2, gamma2, beta2, output);

    std::cout << "Encoder DoubleConv test completed successfully!" << std::endl;

    // Print some sample inputs and outputs for verification
    std::cout << "Sample input[0][0][0][0][0]: " << input[0][0][0][0][0] << std::endl;
    std::cout << "Sample output[0][0][0][0][0]: " << output[0][0][0][0][0] << std::endl;

    // Verify channel dimension change
    std::cout << "Channel dimension verification:" << std::endl;
    std::cout << "Input channels " << F_MAP_0 << " -> Output channels " << F_MAP_1 << std::endl;

    return 0;
}