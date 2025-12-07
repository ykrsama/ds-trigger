#include "test_conv.h"
#include "../template/model.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    // Initialize test data
    data_t kernel[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL];
    data_t input[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    data_t output[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];

    // Seed random number generator
    srand(time(NULL));

    // Initialize input with random values
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < F_MAP_h; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        input[b][c][d][h][w] = data_t((float)rand() / RAND_MAX - 0.5f); // Random values between -0.5 and 0.5
                    }
                }
            }
        }
    }

    // Initialize kernel with small random weights
    for (int out_c = 0; out_c < F_MAP_0; out_c++) {
        for (int in_c = 0; in_c < F_MAP_h; in_c++) {
            for (int d = 0; d < CONV_KERNEL; d++) {
                for (int h = 0; h < CONV_KERNEL; h++) {
                    for (int w = 0; w < CONV_KERNEL; w++) {
                        kernel[out_c][in_c][d][h][w] = data_t(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
                    }
                }
            }
        }
    }

    // Initialize output
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < F_MAP_0; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        output[b][c][d][h][w] = data_t(0);
                    }
                }
            }
        }
    }

    std::cout << "Starting  TestConv test..." << std::endl;
    std::cout << "Input dimensions: [" << BATCH_SIZE << "][" << F_MAP_h << "]["
              << INPUT_DEPTH << "][" << INPUT_HEIGHT << "][" << INPUT_WIDTH << "]" << std::endl;
    std::cout << "Kernel dimensions: [" << F_MAP_0 << "][" << F_MAP_h << "]["
              << CONV_KERNEL << "][" << CONV_KERNEL << "][" << CONV_KERNEL << "]" << std::endl;
    std::cout << "Output dimensions: [" << BATCH_SIZE << "][" << F_MAP_0 << "]["
              << INPUT_DEPTH << "][" << INPUT_HEIGHT << "][" << INPUT_WIDTH << "]" << std::endl;

    // Call the top function
     TestConv(kernel, input, output);

    std::cout << " TestConv test completed successfully!" << std::endl;

    // Print some sample inputs and outputs for verification
    std::cout << "Sample input[0][0][0][0][0]: " << (float)input[0][0][0][0][0] << std::endl;
    std::cout << "Sample kernel[0][0][0][0][0]: " << (float)kernel[0][0][0][0][0] << std::endl;
    std::cout << "Sample output[0][0][0][0][0]: " << (float)output[0][0][0][0][0] << std::endl;

    // Verify channel dimension change
    std::cout << "Channel dimension verification:" << std::endl;
    std::cout << "Input channels " << F_MAP_h << " -> Output channels " << F_MAP_0 << std::endl;

    // Check if output is non-zero (basic sanity check)
    bool has_nonzero = false;
    for (int b = 0; b < BATCH_SIZE && !has_nonzero; b++) {
        for (int c = 0; c < F_MAP_0 && !has_nonzero; c++) {
            for (int d = 0; d < INPUT_DEPTH && !has_nonzero; d++) {
                for (int h = 0; h < INPUT_HEIGHT && !has_nonzero; h++) {
                    for (int w = 0; w < INPUT_WIDTH && !has_nonzero; w++) {
                        if (output[b][c][d][h][w] != data_t(0)) {
                            has_nonzero = true;
                        }
                    }
                }
            }
        }
    }

    if (has_nonzero) {
        std::cout << "PASS: Output contains non-zero values" << std::endl;
    } else {
        std::cout << "WARNING: Output is all zeros - check implementation" << std::endl;
    }

    return 0;
}
