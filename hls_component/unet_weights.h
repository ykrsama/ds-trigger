#ifndef UNET_WEIGHTS_H
#define UNET_WEIGHTS_H

// Auto-generated weight declarations for 3D U-Net
// Reduced contraction path architecture

// Input Convolution Block: 1 -> 64 -> 64 channels
extern float input_conv1_weight[];
extern float input_conv1_gamma[];
extern float input_conv1_beta[];
extern float input_conv2_weight[];
extern float input_conv2_gamma[];
extern float input_conv2_beta[];

// Encoder Convolution Block: 64 -> 128 -> 128 channels
extern float encoder_conv1_weight[];
extern float encoder_conv1_gamma[];
extern float encoder_conv1_beta[];
extern float encoder_conv2_weight[];
extern float encoder_conv2_gamma[];
extern float encoder_conv2_beta[];

// Decoder Convolution Block: 192 -> 64 -> 64 channels
extern float decoder_conv1_weight[];
extern float decoder_conv1_gamma[];
extern float decoder_conv1_beta[];
extern float decoder_conv2_weight[];
extern float decoder_conv2_gamma[];
extern float decoder_conv2_beta[];

// Output Convolution Block: 64 -> 64 -> 64 channels
extern float output_conv1_weight[];
extern float output_conv1_gamma[];
extern float output_conv1_beta[];
extern float output_conv2_weight[];
extern float output_conv2_gamma[];
extern float output_conv2_beta[];

// Final Convolution: 64 -> 5 channels
extern float final_conv_weight[];
extern float final_conv_bias[];

#endif // UNET_WEIGHTS_H
