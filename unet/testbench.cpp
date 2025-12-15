#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include "unet.h"

using namespace std;

// Test data generation utilities
void generate_test_input(data_t input[BATCH_SIZE][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]) {
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int d = 0; d < INPUT_DEPTH; d++) {
            for (int h = 0; h < INPUT_HEIGHT; h++) {
                for (int w = 0; w < INPUT_WIDTH; w++) {
                    // Generate normalized test data
                    float val = (float)(rand() % 1000) / 1000.0f - 0.5f;
                    input[b][d][h][w] = data_t(val);
                }
            }
        }
    }
}

void generate_test_kernels(data_t conv1_kernel1[F_MAP_h][INPUT_DEPTH][CONV_KERNEL][CONV_KERNEL],
                          data_t conv1_kernel2[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL],
                          data_t conv2_kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL],
                          data_t conv2_kernel2[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL],
                          data_t conv3_kernel1[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL],
                          data_t conv3_kernel2[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL],
                          data_t final_kernel[OUT_CHANNELS][F_MAP_0][1][1][1],
                          data_t final_bias[OUT_CHANNELS]) {

    // Initialize conv1_kernel1
    for (int oc = 0; oc < F_MAP_h; oc++) {
        for (int ic = 0; ic < INPUT_DEPTH; ic++) {
            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                    conv1_kernel1[oc][ic][kh][kw] = data_t((float)(rand() % 200 - 100) / 1000.0f);
                }
            }
        }
    }

    // Initialize conv1_kernel2
    for (int oc = 0; oc < F_MAP_0; oc++) {
        for (int ic = 0; ic < F_MAP_h; ic++) {
            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                    conv1_kernel2[oc][ic][kh][kw] = data_t((float)(rand() % 200 - 100) / 1000.0f);
                }
            }
        }
    }

    // Initialize conv2_kernel1
    for (int oc = 0; oc < F_MAP_0; oc++) {
        for (int ic = 0; ic < F_MAP_0; ic++) {
            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                    conv2_kernel1[oc][ic][kh][kw] = data_t((float)(rand() % 200 - 100) / 1000.0f);
                }
            }
        }
    }

    // Initialize conv2_kernel2
    for (int oc = 0; oc < F_MAP_1; oc++) {
        for (int ic = 0; ic < F_MAP_0; ic++) {
            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                    conv2_kernel2[oc][ic][kh][kw] = data_t((float)(rand() % 200 - 100) / 1000.0f);
                }
            }
        }
    }

    // Initialize conv3_kernel1
    for (int oc = 0; oc < F_MAP_0; oc++) {
        for (int ic = 0; ic < CONCAT_CHANNELS; ic++) {
            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                    conv3_kernel1[oc][ic][kh][kw] = data_t((float)(rand() % 200 - 100) / 1000.0f);
                }
            }
        }
    }

    // Initialize conv3_kernel2
    for (int oc = 0; oc < F_MAP_0; oc++) {
        for (int ic = 0; ic < F_MAP_0; ic++) {
            for (int kh = 0; kh < CONV_KERNEL; kh++) {
                for (int kw = 0; kw < CONV_KERNEL; kw++) {
                    conv3_kernel2[oc][ic][kh][kw] = data_t((float)(rand() % 200 - 100) / 1000.0f);
                }
            }
        }
    }

    // Initialize final_kernel
    for (int oc = 0; oc < OUT_CHANNELS; oc++) {
        for (int ic = 0; ic < F_MAP_0; ic++) {
            final_kernel[oc][ic][0][0][0] = data_t((float)(rand() % 200 - 100) / 1000.0f);
        }
    }

    // Initialize final_bias
    for (int oc = 0; oc < OUT_CHANNELS; oc++) {
        final_bias[oc] = data_t((float)(rand() % 200 - 100) / 1000.0f);
    }
}

void print_output_sample(data_t output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]) {
    cout << "Sample output values:" << endl;
    for (int c = 0; c < OUT_CHANNELS && c < 3; c++) {
        cout << "Channel " << c << ":" << endl;
        for (int d = 0; d < INPUT_DEPTH && d < 3; d++) {
            cout << "  Depth " << d << ": ";
            for (int h = 0; h < INPUT_HEIGHT && h < 3; h++) {
                for (int w = 0; w < INPUT_WIDTH && w < 3; w++) {
                    cout << (float)output[0][c][d][h][w] << " ";
                }
                if (INPUT_WIDTH > 3) cout << "... ";
            }
            cout << endl;
            if (INPUT_HEIGHT > 3) cout << "  ..." << endl;
        }
        if (INPUT_DEPTH > 3) cout << "  ..." << endl;
        cout << endl;
    }
    if (OUT_CHANNELS > 3) cout << "..." << endl;
}

int main() {
    cout << "=== UNet Testbench ===" << endl;
    cout << "Network Configuration:" << endl;
    cout << "  Input size: " << BATCH_SIZE << "x" << INPUT_DEPTH << "x" << INPUT_HEIGHT << "x" << INPUT_WIDTH << endl;
    cout << "  Output size: " << BATCH_SIZE << "x" << OUT_CHANNELS << "x" << INPUT_DEPTH << "x" << INPUT_HEIGHT << "x" << INPUT_WIDTH << endl;
    cout << "  F_MAP_0: " << F_MAP_0 << ", F_MAP_1: " << F_MAP_1 << ", F_MAP_h: " << F_MAP_h << endl;
    cout << "  CONCAT_CHANNELS: " << CONCAT_CHANNELS << endl;
    cout << endl;

    // Allocate memory for test data
    static data_t input[BATCH_SIZE][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    static data_t output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];

    // Allocate kernel arrays
    static data_t conv1_kernel1[F_MAP_h][INPUT_DEPTH][CONV_KERNEL][CONV_KERNEL];
    static data_t conv1_kernel2[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL];
    static data_t conv2_kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL];
    static data_t conv2_kernel2[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL];
    static data_t conv3_kernel1[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL];
    static data_t conv3_kernel2[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL];
    static data_t final_kernel[OUT_CHANNELS][F_MAP_0][1][1][1];
    static data_t final_bias[OUT_CHANNELS];

    // Seed random number generator
    srand(12345);

    cout << "Generating test data..." << endl;
    generate_test_input(input);
    generate_test_kernels(conv1_kernel1, conv1_kernel2, conv2_kernel1, conv2_kernel2,
                         conv3_kernel1, conv3_kernel2, final_kernel, final_bias);

    cout << "Running UNet inference..." << endl;

    // Run the UNet
    UNet(input, conv1_kernel1, conv1_kernel2, conv2_kernel1, conv2_kernel2,
         conv3_kernel1, conv3_kernel2, final_kernel, final_bias, output);

    cout << "Inference completed!" << endl;
    cout << endl;

    // Print sample outputs
    print_output_sample(output);

    // Basic output validation
    bool has_nonzero = false;
    bool has_nan = false;
    data_t sum = 0;
    int count = 0;

    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int c = 0; c < OUT_CHANNELS; c++) {
            for (int d = 0; d < INPUT_DEPTH; d++) {
                for (int h = 0; h < INPUT_HEIGHT; h++) {
                    for (int w = 0; w < INPUT_WIDTH; w++) {
                        data_t val = output[b][c][d][h][w];
                        if (val != 0.0) has_nonzero = true;
                        if (isnan((float)val)) has_nan = true;
                        sum += val;
                        count++;
                    }
                }
            }
        }
    }

    data_t mean = sum / count;

    cout << "Output Statistics:" << endl;
    cout << "  Has non-zero values: " << (has_nonzero ? "YES" : "NO") << endl;
    cout << "  Has NaN values: " << (has_nan ? "YES" : "NO") << endl;
    cout << "  Mean value: " << (float)mean << endl;
    cout << "  Total elements: " << count << endl;
    cout << endl;

    // Check if test passed
    if (!has_nan && has_nonzero) {
        cout << "*** TEST PASSED ***" << endl;
        return 0;
    } else {
        cout << "*** TEST FAILED ***" << endl;
        if (has_nan) cout << "  Reason: Output contains NaN values" << endl;
        if (!has_nonzero) cout << "  Reason: All output values are zero" << endl;
        return 1;
    }
}