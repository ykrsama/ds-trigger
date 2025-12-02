#ifndef UNET_WEIGHTS_H
#define UNET_WEIGHTS_H

// Auto-generated weight declarations for 3D U-Net
// Reduced contraction path architecture

// Input Convolution Block: 1 -> 64 -> 64 channels
extern float input_conv1_weight[32][1][3][3][3];
extern float input_conv2_weight[64][32][3][3][3];
// Encoder Convolution Block: 64 -> 128 -> 128 channels
extern float encoder_conv1_weight[64][64][3][3][3];
extern float encoder_conv2_weight[128][64][3][3][3];
// Decoder Convolution Block: 192 -> 64 -> 64 channels
extern float decoder_conv1_weight[64][192][3][3][3];
extern float decoder_conv2_weight[64][64][3][3][3];
// Output Convolution Block: 64 -> 64 -> 64 channels
extern float output_conv1_weight[64][64][3][3][3];
extern float output_conv2_weight[64][64][3][3][3];
// Final Convolution: 64 -> 5 channels
extern float final_conv_weight[5][64][1][1][1];
extern float final_conv_bias[5];

#endif // UNET_WEIGHTS_H
