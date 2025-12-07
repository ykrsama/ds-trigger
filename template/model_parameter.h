#ifndef MODEL_PARAMETER_H
#define MODEL_PARAMETER_H

// Network configuration constants
#define BATCH_SIZE 1
#define IN_CHANNELS 1
#define OUT_CHANNELS 5
#define F_MAP_0 32
#define F_MAP_1 64

// Input dimensions (configurable based on dataset)
#define INPUT_DEPTH 43
#define INPUT_HEIGHT 43
#define INPUT_WIDTH 11

// Convolution parameters
#define CONV_KERNEL 3
#define CONV_PADDING 1
#define CONV_STRIDE 1

// Pooling parameters
#define POOL_KERNEL 2
#define POOL_STRIDE 2

#define CONV_OUTPUT_DEPTH ((INPUT_DEPTH + 2 * CONV_PADDING - CONV_KERNEL) / CONV_STRIDE + 1)
#define CONV_OUTPUT_HEIGHT ((INPUT_HEIGHT + 2 * CONV_PADDING - CONV_KERNEL) / CONV_STRIDE + 1)
#define CONV_OUTPUT_WIDTH ((INPUT_WIDTH + 2 * CONV_PADDING - CONV_KERNEL) / CONV_STRIDE + 1)

#define POOL_OUTPUT_DEPTH (CONV_OUTPUT_DEPTH / POOL_STRIDE)
#define POOL_OUTPUT_HEIGHT (CONV_OUTPUT_HEIGHT / POOL_STRIDE)
#define POOL_OUTPUT_WIDTH (CONV_OUTPUT_WIDTH / POOL_STRIDE)

#define UPSAMPLE_OUTPUT_DEPTH (POOL_OUTPUT_DEPTH * 2)
#define UPSAMPLE_OUTPUT_HEIGHT (POOL_OUTPUT_HEIGHT * 2)
#define UPSAMPLE_OUTPUT_WIDTH (POOL_OUTPUT_WIDTH * 2)

#define CONCAT_CHANNELS (F_MAP_0 + F_MAP_1)
#define F_MAP_h (F_MAP_0 / 2)  // Half of F_MAP_0 for input conv block

// GroupNorm parameters
#define NUM_GROUPS 8  // choose 1 here because num out channel is 5
#define EPSILON 1e-5f

#endif // MODEL_PARAMETER_H
