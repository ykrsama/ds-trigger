#include "include/unet3d_reduced.h"
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <cmath>

using namespace std;

// Testbench for UNet3DReduced HLS implementation
int main() {
    cout << "Starting UNet3D Reduced HLS Testbench..." << endl;

    // Initialize random number generator
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> normal_dist(0.0f, 1.0f);
    uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);

    // Allocate input and output tensors
    cout << "Allocating memory for input/output tensors..." << endl;

    // Input tensor
    static float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];

    // Output tensor
    static float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];

    cout << "Memory allocation complete. Tensor sizes:" << endl;
    cout << "  Input: [" << BATCH_SIZE << ", " << INPUT_CHANNELS << ", "
         << INPUT_DEPTH << ", " << INPUT_HEIGHT << ", " << INPUT_WIDTH << "]" << endl;
    cout << "  Output: [" << BATCH_SIZE << ", " << OUTPUT_CHANNELS << ", "
         << INPUT_DEPTH << ", " << INPUT_HEIGHT << ", " << INPUT_WIDTH << "]" << endl;

    // Initialize input data with random values
    cout << "Initializing input data with random values..." << endl;
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < INPUT_CHANNELS; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        input[b][c][d][h][w] = uniform_dist(gen);
                    }
                }
            }
        }
    }


    // Save input data to file for verification
    cout << "Saving input data to 'input_data.txt'..." << endl;
    ofstream input_file("input_data.txt");
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < INPUT_CHANNELS; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        input_file << input[b][c][d][h][w] << " ";
                    }
                    input_file << endl;
                }
            }
        }
    }
    input_file.close();

    // Run the UNet3D function
    cout << "Running UNet3D HLS function..." << endl;
    auto start_time = chrono::high_resolution_clock::now();

    UNet3DReduced(
        input,
        input_conv1_weight,
        input_conv2_weight,
        encoder_conv1_weight,
        encoder_conv2_weight,
        decoder_conv1_weight,
        decoder_conv2_weight,
        output_conv1_weight,
        output_conv2_weight,
        final_conv_weight,
        final_conv_bias,
        input_conv1_gamma,
        input_conv1_beta,
        input_conv2_gamma,
        input_conv2_beta,
        encoder_conv1_gamma,
        encoder_conv1_beta,
        encoder_conv2_gamma,
        encoder_conv2_beta,
        decoder_conv1_gamma,
        decoder_conv1_beta,
        decoder_conv2_gamma,
        decoder_conv2_beta,
        output_conv1_gamma,
        output_conv1_beta,
        output_conv2_gamma,
        output_conv2_beta,
        output
    );

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "UNet3D execution completed in " << duration.count() << " ms" << endl;

    // Verify output
    cout << "Verifying output..." << endl;

    // Check for valid output range (sigmoid outputs should be between 0 and 1)
    float min_val = 1e9f, max_val = -1e9f;
    float sum_val = 0.0f;
    int total_elements = BATCH_SIZE * OUTPUT_CHANNELS * INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH;
    int nan_count = 0, inf_count = 0;

    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < OUTPUT_CHANNELS; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        float val = output[b][c][d][h][w];

                        if (isnan(val)) {
                            nan_count++;
                        } else if (isinf(val)) {
                            inf_count++;
                        } else {
                            if (val < min_val) min_val = val;
                            if (val > max_val) max_val = val;
                            sum_val += val;
                        }
                    }
                }
            }
        }
    }

    float mean_val = sum_val / (total_elements - nan_count - inf_count);

    cout << "Output statistics:" << endl;
    cout << "  Min value: " << min_val << endl;
    cout << "  Max value: " << max_val << endl;
    cout << "  Mean value: " << mean_val << endl;
    cout << "  NaN count: " << nan_count << endl;
    cout << "  Inf count: " << inf_count << endl;
    cout << "  Total elements: " << total_elements << endl;

    // Save output data to file
    cout << "Saving output data to 'output_data.txt'..." << endl;
    ofstream output_file("output_data.txt");
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < OUTPUT_CHANNELS; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        output_file << output[b][c][d][h][w] << " ";
                    }
                    output_file << endl;
                }
            }
        }
    }
    output_file.close();

    // Sample output values for manual verification
    cout << "Sample output values:" << endl;
    for (int i = 0; i < min(5, OUTPUT_CHANNELS); i++) {
        cout << "  Channel " << i << " at (0,0,0): "
             << output[0][i][0][0][0] << endl;
    }

    // Test individual blocks for debugging
    cout << "\nTesting individual blocks..." << endl;

    // Test input convolution
    static float test_input_conv_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    cout << "Testing InputDoubleConv3D..." << endl;
    InputDoubleConv3D(input, input_conv1_weight, input_conv1_gamma, input_conv1_beta,
                      input_conv2_weight, input_conv2_gamma, input_conv2_beta, test_input_conv_out);

    float input_conv_min = 1e9f, input_conv_max = -1e9f;
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < min(5, F_MAP_0); c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        float val = test_input_conv_out[b][c][d][h][w];
                        if (val < input_conv_min) input_conv_min = val;
                        if (val > input_conv_max) input_conv_max = val;
                    }
                }
            }
        }
    }
    cout << "  InputDoubleConv3D output range: [" << input_conv_min << ", " << input_conv_max << "]" << endl;

    // Validation checks
    bool test_passed = true;

    if (nan_count > 0) {
        cout << "ERROR: Found NaN values in output!" << endl;
        test_passed = false;
    }

    if (inf_count > 0) {
        cout << "ERROR: Found infinite values in output!" << endl;
        test_passed = false;
    }

    // For sigmoid output, values should be between 0 and 1
    if (min_val < -0.1f || max_val > 1.1f) {
        cout << "WARNING: Output values outside expected sigmoid range [0, 1]" << endl;
        cout << "  This might indicate numerical issues or incorrect implementation" << endl;
    }

    cout << "\n=== Testbench Results ===" << endl;
    cout << "Test Status: " << (test_passed ? "PASSED" : "FAILED") << endl;
    cout << "Execution Time: " << duration.count() << " ms" << endl;
    cout << "Output files generated: input_data.txt, output_data.txt" << endl;

    if (test_passed) {
        cout << "\nAll tests passed! The UNet3D HLS implementation appears to be working correctly." << endl;
        cout << "You can now proceed with HLS synthesis." << endl;
    } else {
        cout << "\nSome tests failed. Please check the implementation for numerical stability issues." << endl;
    }

    return test_passed ? 0 : 1;
}