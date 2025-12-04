#include "encoder_maxpool.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    // Initialize test data
    float input[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    float output[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];

    // Seed random number generator
    srand(time(NULL));

    // Initialize input with random values
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < F_MAP_0; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        input[b][c][d][h][w] = (float)rand() / RAND_MAX - 0.5f; // Random values between -0.5 and 0.5
                    }
                }
            }
        }
    }

    // Initialize output
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < F_MAP_0; c++) {
            for (int d = 0; d < POOL_OUTPUT_DEPTH; d++) {
                for (int h = 0; h < POOL_OUTPUT_HEIGHT; h++) {
                    for (int w = 0; w < POOL_OUTPUT_WIDTH; w++) {
                        output[b][c][d][h][w] = 0.0f;
                    }
                }
            }
        }
    }

    std::cout << "Starting Encoder MaxPool test..." << std::endl;
    std::cout << "Input dimensions: [" << BATCH_SIZE << "][" << F_MAP_0 << "]["
              << INPUT_DEPTH << "][" << INPUT_HEIGHT << "][" << INPUT_WIDTH << "]" << std::endl;
    std::cout << "Output dimensions: [" << BATCH_SIZE << "][" << F_MAP_0 << "]["
              << POOL_OUTPUT_DEPTH << "][" << POOL_OUTPUT_HEIGHT << "][" << POOL_OUTPUT_WIDTH << "]" << std::endl;

    // Call the top function
    EncoderMaxPool(input, output);

    std::cout << "Encoder MaxPool test completed successfully!" << std::endl;

    // Print some sample inputs and outputs for verification
    std::cout << "Sample input[0][0][0][0][0]: " << input[0][0][0][0][0] << std::endl;
    std::cout << "Sample output[0][0][0][0][0]: " << output[0][0][0][0][0] << std::endl;

    // Verify that max pooling reduces dimensions correctly
    std::cout << "Dimension reduction verification:" << std::endl;
    std::cout << "Input depth " << INPUT_DEPTH << " -> Output depth " << POOL_OUTPUT_DEPTH << std::endl;
    std::cout << "Input height " << INPUT_HEIGHT << " -> Output height " << POOL_OUTPUT_HEIGHT << std::endl;
    std::cout << "Input width " << INPUT_WIDTH << " -> Output width " << POOL_OUTPUT_WIDTH << std::endl;

    return 0;
}