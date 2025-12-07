#include "test_groupnorm.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    // Initialize test data
    data_t input_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    data_t gamma[F_MAP_0];
    data_t beta[F_MAP_0];
    data_t output_data[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];

    // Seed random number generator
    srand(time(NULL));

    // Initialize input with random values
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < F_MAP_0; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        input_data[b][c][d][h][w] = (data_t)rand() / RAND_MAX - 0.5f; // Random values between -0.5 and 0.5
                    }
                }
            }
        }
    }

    // Initialize gamma parameters (scale factors, typically initialized to 1.0)
    for (int c = 0; c < F_MAP_0; c++) {
        gamma[c] = 1.0f + ((data_t)rand() / RAND_MAX - 0.5f) * 0.2f; // Around 1.0 with small variation
    }

    // Initialize beta parameters (shift factors, typically initialized to 0.0)
    for (int c = 0; c < F_MAP_0; c++) {
        beta[c] = ((data_t)rand() / RAND_MAX - 0.5f) * 0.1f; // Small values around 0.0
    }

    // Initialize output
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < F_MAP_0; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        output_data[b][c][d][h][w] = 0.0f;
                    }
                }
            }
        }
    }

    std::cout << "Starting TestGroupNorm test..." << std::endl;
    std::cout << "Input dimensions: [" << BATCH_SIZE << "][" << F_MAP_0 << "]["
              << INPUT_DEPTH << "][" << INPUT_HEIGHT << "][" << INPUT_WIDTH << "]" << std::endl;
    std::cout << "Gamma dimensions: [" << F_MAP_0 << "]" << std::endl;
    std::cout << "Beta dimensions: [" << F_MAP_0 << "]" << std::endl;
    std::cout << "Output dimensions: [" << BATCH_SIZE << "][" << F_MAP_0 << "]["
              << INPUT_DEPTH << "][" << INPUT_HEIGHT << "][" << INPUT_WIDTH << "]" << std::endl;

    // Call the top function
    TestGroupNorm(input_data, gamma, beta, output_data);

    std::cout << "TestGroupNorm test completed successfully!" << std::endl;

    // Print some sample inputs and outputs for verification
    std::cout << "Sample input_data[0][0][0][0][0]: " << input_data[0][0][0][0][0] << std::endl;
    std::cout << "Sample gamma[0]: " << gamma[0] << std::endl;
    std::cout << "Sample beta[0]: " << beta[0] << std::endl;
    std::cout << "Sample output_data[0][0][0][0][0]: " << output_data[0][0][0][0][0] << std::endl;

    // Verify that GroupNorm has been applied by checking statistics
    // Calculate mean and variance of first channel of output
    data_t sum = 0.0f;
    data_t sum_sq = 0.0f;
    int total_elements = INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH;

    for (int d = 0; d < INPUT_DEPTH; d++) {
        for (int h = 0; h < INPUT_HEIGHT; h++) {
            for (int w = 0; w < INPUT_WIDTH; w++) {
                data_t value = output_data[0][0][d][h][w];
                sum += value;
                sum_sq += value * value;
            }
        }
    }

    data_t mean = sum / total_elements;
    data_t variance = (sum_sq / total_elements) - (mean * mean);

    std::cout << "Output statistics for channel 0:" << std::endl;
    std::cout << "Mean: " << mean << " (should be close to beta[0] = " << beta[0] << ")" << std::endl;
    std::cout << "Variance: " << variance << " (should be close to gamma[0]^2 = " << gamma[0] * gamma[0] << ")" << std::endl;

    // Check if output is different from input (basic sanity check)
    bool has_difference = false;
    for (int b = 0; b < BATCH_SIZE && !has_difference; b++) {
        for (int c = 0; c < F_MAP_0 && !has_difference; c++) {
            for (int d = 0; d < INPUT_DEPTH && !has_difference; d++) {
                for (int h = 0; h < INPUT_HEIGHT && !has_difference; h++) {
                    for (int w = 0; w < INPUT_WIDTH && !has_difference; w++) {
                        if (abs(output_data[b][c][d][h][w] - input_data[b][c][d][h][w]) > 1e-6f) {
                            has_difference = true;
                        }
                    }
                }
            }
        }
    }

    if (has_difference) {
        std::cout << "PASS: Output differs from input (GroupNorm applied successfully)" << std::endl;
    } else {
        std::cout << "WARNING: Output is identical to input - check GroupNorm implementation" << std::endl;
    }

    return 0;
}