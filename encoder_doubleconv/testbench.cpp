#include "encoder_doubleconv.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    // Initialize test data
    data_t input[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    data_t kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    data_t gamma1[F_MAP_0];
    data_t beta1[F_MAP_0];
    data_t kernel2[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    data_t gamma2[F_MAP_0];
    data_t beta2[F_MAP_0];
    data_t output[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];

    // Seed random number generator
    srand(time(NULL));

    // Initialize input with random values
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < F_MAP_0; c++) {
            for (int d = 0; d < POOL_OUTPUT_DEPTH; d++) {
                for (int h = 0; h < POOL_OUTPUT_HEIGHT; h++) {
                    for (int w = 0; w < POOL_OUTPUT_WIDTH; w++) {
                        input[b][c][d][h][w] = (data_t)rand() / RAND_MAX - (data_t)0.5; // Random values between -0.5 and 0.5
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
                        kernel1[out_c][in_c][d][h][w] = ((data_t)rand() / RAND_MAX - (data_t)0.5) * (data_t)0.1;
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
                        kernel2[out_c][in_c][d][h][w] = ((data_t)rand() / RAND_MAX - (data_t)0.5) * (data_t)0.1;
                    }
                }
            }
        }
    }

    // Initialize group norm parameters
    for (int c = 0; c < F_MAP_0; c++) {
        gamma1[c] = (data_t)1.0;
        beta1[c]  = (data_t)0.0;
        gamma2[c] = (data_t)1.0;
        beta2[c]  = (data_t)0.0;
    }

    // Initialize output
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < F_MAP_1; c++) {
            for (int d = 0; d < POOL_OUTPUT_DEPTH; d++) {
                for (int h = 0; h < POOL_OUTPUT_HEIGHT; h++) {
                    for (int w = 0; w < POOL_OUTPUT_WIDTH; w++) {
                        output[b][c][d][h][w] = (data_t)0.0;
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
