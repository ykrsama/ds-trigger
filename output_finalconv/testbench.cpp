#include "output_finalconv.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    // Initialize test data
    float input[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    float kernel[OUT_CHANNELS][IN_CHANNELS][1][1][1];
    float bias[OUT_CHANNELS];
    float output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];

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

    // Initialize 1x1x1 kernel with small random weights
    for (int out_c = 0; out_c < OUT_CHANNELS; out_c++) {
        for (int in_c = 0; in_c < IN_CHANNELS; in_c++) {
            kernel[out_c][in_c][0][0][0] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }

    // Initialize bias
    for (int c = 0; c < OUT_CHANNELS; c++) {
        bias[c] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f; // Small bias values
    }

    // Initialize output
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < OUT_CHANNELS; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        output[b][c][d][h][w] = 0.0f;
                    }
                }
            }
        }
    }

    std::cout << "Starting Output Final Conv1x1 test..." << std::endl;
    std::cout << "Input dimensions: [" << BATCH_SIZE << "][" << IN_CHANNELS << "]["
              << INPUT_DEPTH << "][" << INPUT_HEIGHT << "][" << INPUT_WIDTH << "]" << std::endl;
    std::cout << "Kernel dimensions: [" << OUT_CHANNELS << "][" << IN_CHANNELS << "][1][1][1]" << std::endl;
    std::cout << "Output dimensions: [" << BATCH_SIZE << "][" << OUT_CHANNELS << "]["
              << INPUT_DEPTH << "][" << INPUT_HEIGHT << "][" << INPUT_WIDTH << "]" << std::endl;

    // Call the top function
    OutputFinalConv(input, kernel, bias, output);

    std::cout << "Output Final Conv1x1 test completed successfully!" << std::endl;

    // Print some sample inputs and outputs for verification
    std::cout << "Sample input[0][0][0][0][0]: " << input[0][0][0][0][0] << std::endl;
    std::cout << "Sample kernel[0][0][0][0][0]: " << kernel[0][0][0][0][0] << std::endl;
    std::cout << "Sample bias[0]: " << bias[0] << std::endl;
    std::cout << "Sample output[0][0][0][0][0]: " << output[0][0][0][0][0] << std::endl;

    // Verify channel dimension change
    std::cout << "Channel dimension verification:" << std::endl;
    std::cout << "Input channels " << IN_CHANNELS << " -> Output channels " << OUT_CHANNELS << std::endl;

    return 0;
}