#include "unet3d_reduced.h"

// Top-level UNet3D function
void UNet3DReduced(
    // Input
    float input[BATCH_SIZE][INPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],

    // Weights for all layers
    float input_conv1_weight[F_MAP_h][INPUT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input_conv2_weight[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float encoder_conv1_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float encoder_conv2_weight[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float decoder_conv1_weight[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float decoder_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float output_conv1_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float output_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float final_conv_weight[OUTPUT_CHANNELS][F_MAP_0][1][1][1],
    float final_conv_bias[OUTPUT_CHANNELS],

    // GroupNorm parameters
    float input_conv1_gamma[INPUT_CHANNELS],
    float input_conv1_beta[INPUT_CHANNELS],
    float input_conv2_gamma[F_MAP_h],
    float input_conv2_beta[F_MAP_h],

    float encoder_conv1_gamma[F_MAP_0],
    float encoder_conv1_beta[F_MAP_0],
    float encoder_conv2_gamma[F_MAP_0],
    float encoder_conv2_beta[F_MAP_0],

    float decoder_conv1_gamma[CONCAT_CHANNELS],
    float decoder_conv1_beta[CONCAT_CHANNELS],
    float decoder_conv2_gamma[F_MAP_0],
    float decoder_conv2_beta[F_MAP_0],

    float output_conv1_gamma[F_MAP_0],
    float output_conv1_beta[F_MAP_0],
    float output_conv2_gamma[F_MAP_0],
    float output_conv2_beta[F_MAP_0],

    // Output
    float output[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return
    #pragma HLS interface bram port=input
    #pragma HLS interface bram port=input_conv1_weight
    #pragma HLS interface bram port=input_conv2_weight
    #pragma HLS interface bram port=encoder_conv1_weight
    #pragma HLS interface bram port=encoder_conv2_weight
    #pragma HLS interface bram port=decoder_conv1_weight
    #pragma HLS interface bram port=decoder_conv2_weight
    #pragma HLS interface bram port=output_conv1_weight
    #pragma HLS interface bram port=output_conv2_weight
    #pragma HLS interface bram port=final_conv_weight
    #pragma HLS interface bram port=final_conv_bias
    #pragma HLS interface bram port=output

    #pragma HLS dataflow

    // Intermediate buffers for dataflow

    // Input path buffers
    float input_conv1_out[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=input_conv1_out depth=10 type=fifo

    float input_conv1_out[BATCH_SIZE][F_MAP_h][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=input_conv1_out depth=10 type=fifo

    float input_conv2_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=input_conv2_out depth=10 type=fifo

    float input_conv2_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=input_conv2_out depth=10 type=fifo

    // Encoder path buffers
    float encoder_pool_out[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_pool_out depth=10 type=fifo

    float encoder_conv1_out[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_conv1_out depth=10 type=fifo

    float encoder_conv1_out[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_conv1_out depth=10 type=fifo

    float encoder_conv2_out[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_conv2_out depth=10 type=fifo

    float encoder_conv2_out[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_conv2_out depth=10 type=fifo

    // Decoder path buffers
    float decoder_upsample_out[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_upsample_out depth=10 type=fifo

    float concat_out[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=concat_out depth=10 type=fifo

    float decoder_conv1_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_conv1_out depth=10 type=fifo

    float decoder_conv1_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_conv1_out depth=10 type=fifo

    float decoder_conv2_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_conv2_out depth=10 type=fifo

    float decoder_conv2_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_conv2_out depth=10 type=fifo

    // Output path buffers
    float output_conv1_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=output_conv1_out depth=10 type=fifo

    float output_conv1_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=output_conv1_out depth=10 type=fifo

    float output_conv2_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=output_conv2_out depth=10 type=fifo

    float output_conv2_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=output_conv2_out depth=10 type=fifo

    float final_conv_out[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=final_conv_out depth=10 type=fifo

    // UNet forward pass
    // Input path: Input → InputConv3D_1 → InputGroupNorm3D_1 → InputConv3D_2 → InputGroupNorm3D_2
    InputConv3D_1(input_conv1_weight, input, input_conv1_out);
    InputGroupNorm3D_1(input_conv1_out, input_conv1_gamma, input_conv1_beta, input_conv1_out);
    InputConv3D_2(input_conv2_weight, input_conv1_out, input_conv2_out);
    InputGroupNorm3D_2(input_conv2_out, input_conv2_gamma, input_conv2_beta, input_conv2_out);

    // Encoder path: EncoderMaxPool3D → EncoderConv3D_1 → EncoderGroupNorm3D_1 → EncoderConv3D_2 → EncoderGroupNorm3D_2
    EncoderMaxPool3D(input_conv2_out, encoder_pool_out);
    EncoderConv3D_1(encoder_conv1_weight, encoder_pool_out, encoder_conv1_out);
    EncoderGroupNorm3D_1(encoder_conv1_out, encoder_conv1_gamma, encoder_conv1_beta, encoder_conv1_out);
    EncoderConv3D_2(encoder_conv2_weight, encoder_conv1_out, encoder_conv2_out);
    EncoderGroupNorm3D_2(encoder_conv2_out, encoder_conv2_gamma, encoder_conv2_beta, encoder_conv2_out);

    // Decoder path: DecoderUpsample3D → ConcatenateTensors → DecoderConv3D_1 → DecoderGroupNorm3D_1 → DecoderConv3D_2 → DecoderGroupNorm3D_2
    DecoderUpsample3D(encoder_conv2_out, decoder_upsample_out);
    ConcatenateTensors(input_conv2_out, decoder_upsample_out, concat_out);
    DecoderConv3D_1(decoder_conv1_weight, concat_out, decoder_conv1_out);
    DecoderGroupNorm3D_1(decoder_conv1_out, decoder_conv1_gamma, decoder_conv1_beta, decoder_conv1_out);
    DecoderConv3D_2(decoder_conv2_weight, decoder_conv1_out, decoder_conv2_out);
    DecoderGroupNorm3D_2(decoder_conv2_out, decoder_conv2_gamma, decoder_conv2_beta, decoder_conv2_out);

    // Output path: OutputConv3D_1 → OutputGroupNorm3D_1 → OutputConv3D_2 → OutputGroupNorm3D_2 → FinalConv1x1 → Sigmoid3D
    OutputConv3D_1(output_conv1_weight, decoder_conv2_out, output_conv1_out);
    OutputGroupNorm3D_1(output_conv1_out, output_conv1_gamma, output_conv1_beta, output_conv1_out);
    OutputConv3D_2(output_conv2_weight, output_conv1_out, output_conv2_out);
    OutputGroupNorm3D_2(output_conv2_out, output_conv2_gamma, output_conv2_beta, output_conv2_out);
    FinalConv1x1(final_conv_weight, final_conv_bias, output_conv2_out, final_conv_out);
    Sigmoid3D(final_conv_out, output);
}