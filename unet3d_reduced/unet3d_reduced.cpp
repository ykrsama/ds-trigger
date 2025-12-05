#include "unet3d_reduced.h"

// Top-level UNet3D function
void UNet3DReduced(
    // Input
    float input[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],

    // Weights for all layers
    float input_conv1_weight[F_MAP_h][IN_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float input_conv2_weight[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float encoder_conv1_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float encoder_conv2_weight[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float decoder_conv1_weight[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float decoder_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float output_conv1_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],
    float output_conv2_weight[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL][CONV_KERNEL],

    float final_conv_weight[OUT_CHANNELS][F_MAP_0][1][1][1],
    float final_conv_bias[OUT_CHANNELS],

    // GroupNorm parameters
    float input_conv1_gamma[IN_CHANNELS],
    float input_conv1_beta[IN_CHANNELS],
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
    float output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]
) {
    #pragma HLS interface s_axilite port=return

    #pragma HLS interface bram port=input
    #pragma HLS bind_storage variable=input type=ram_t2p impl=bram

    #pragma HLS interface bram port=input_conv1_weight
    #pragma HLS bind_storage variable=input_conv1_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=input_conv2_weight
    #pragma HLS bind_storage variable=input_conv2_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=encoder_conv1_weight
    #pragma HLS bind_storage variable=encoder_conv1_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=encoder_conv2_weight
    #pragma HLS bind_storage variable=encoder_conv2_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=decoder_conv1_weight
    #pragma HLS bind_storage variable=decoder_conv1_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=decoder_conv2_weight
    #pragma HLS bind_storage variable=decoder_conv2_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=output_conv1_weight
    #pragma HLS bind_storage variable=output_conv1_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=output_conv2_weight
    #pragma HLS bind_storage variable=output_conv2_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=final_conv_weight
    #pragma HLS bind_storage variable=final_conv_weight type=ram_t2p impl=bram

    #pragma HLS interface bram port=final_conv_bias
    #pragma HLS bind_storage variable=final_conv_bias type=ram_t2p impl=bram

    #pragma HLS interface bram port=input_conv1_gamma
    #pragma HLS bind_storage variable=input_conv1_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=input_conv1_beta
    #pragma HLS bind_storage variable=input_conv1_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=input_conv2_gamma
    #pragma HLS bind_storage variable=input_conv2_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=input_conv2_beta
    #pragma HLS bind_storage variable=input_conv2_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=encoder_conv1_gamma
    #pragma HLS bind_storage variable=encoder_conv1_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=encoder_conv1_beta
    #pragma HLS bind_storage variable=encoder_conv1_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=encoder_conv2_gamma
    #pragma HLS bind_storage variable=encoder_conv2_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=encoder_conv2_beta
    #pragma HLS bind_storage variable=encoder_conv2_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=decoder_conv1_gamma
    #pragma HLS bind_storage variable=decoder_conv1_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=decoder_conv1_beta
    #pragma HLS bind_storage variable=decoder_conv1_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=decoder_conv2_gamma
    #pragma HLS bind_storage variable=decoder_conv2_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=decoder_conv2_beta
    #pragma HLS bind_storage variable=decoder_conv2_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=output_conv1_gamma
    #pragma HLS bind_storage variable=output_conv1_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=output_conv1_beta
    #pragma HLS bind_storage variable=output_conv1_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=output_conv2_gamma
    #pragma HLS bind_storage variable=output_conv2_gamma type=ram_t2p impl=bram

    #pragma HLS interface bram port=output_conv2_beta
    #pragma HLS bind_storage variable=output_conv2_beta type=ram_t2p impl=bram

    #pragma HLS interface bram port=output
    #pragma HLS bind_storage variable=output type=ram_t2p impl=bram

    #pragma HLS dataflow

    // Input path buffers - need two streams for skip connection
    float input_conv_out_main[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=input_conv_out_main depth=10 type=fifo
    #pragma HLS bind_storage variable=input_conv_out_main type=ram_2p impl=bram

    float input_conv_out_skip[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=input_conv_out_skip depth=10 type=fifo
    #pragma HLS bind_storage variable=input_conv_out_skip type=ram_2p impl=bram

    // Encoder path buffers
    float encoder_pool_out[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_pool_out depth=10 type=fifo
    #pragma HLS bind_storage variable=encoder_pool_out type=ram_2p impl=bram

    float encoder_conv_out[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=encoder_conv_out type=ram_2p impl=bram

    // Decoder path buffers
    float decoder_upsample_out[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_upsample_out depth=10 type=fifo
    #pragma HLS bind_storage variable=decoder_upsample_out type=ram_2p impl=bram

    float concat_out[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=concat_out depth=10 type=fifo
    #pragma HLS bind_storage variable=concat_out type=ram_2p impl=bram

    float decoder_conv_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=decoder_conv_out type=ram_2p impl=bram

    // Output path buffers
    float output_conv_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=output_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=output_conv_out type=ram_2p impl=bram

    float final_conv_out[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=final_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=final_conv_out type=ram_2p impl=bram

    // UNet forward pass
    // Input Path: [IN_CHANNELS] -> DoubleConv -> [F_MAP_0] (split into two streams)
    DoubleConv3D2Head<IN_CHANNELS, F_MAP_h, F_MAP_0>(
        input,
        input_conv1_weight, input_conv1_gamma, input_conv1_beta,
        input_conv2_weight, input_conv2_gamma, input_conv2_beta,
        input_conv_out_main,
        input_conv_out_skip
    );

    // Encoder Path: [F_MAP_0] -> MaxPool -> [F_MAP_0] -> DoubleConv -> [F_MAP_1]
    MaxPool3D<F_MAP_0>(input_conv_out_main, encoder_pool_out);

    DoubleConv3D<F_MAP_0, F_MAP_0, F_MAP_1, POOL_OUTPUT_DEPTH, POOL_OUTPUT_HEIGHT, POOL_OUTPUT_WIDTH>(
        encoder_pool_out,
        encoder_conv1_weight, encoder_conv1_gamma, encoder_conv1_beta,
        encoder_conv2_weight, encoder_conv2_gamma, encoder_conv2_beta,
        encoder_conv_out
    );

    // [F_MAP_1] -> Upsampling -> [F_MAP_1] -> Concat -> [CONCAT_CHANNELS]
    Upsample3D<F_MAP_1>(encoder_conv_out, decoder_upsample_out);

    ConcatenateTensors<F_MAP_0, F_MAP_1>(input_conv_out_skip, decoder_upsample_out, concat_out);

    DoubleConv3D<CONCAT_CHANNELS, F_MAP_0, F_MAP_0>(
        concat_out,
        decoder_conv1_weight, decoder_conv1_gamma, decoder_conv1_beta,
        decoder_conv2_weight, decoder_conv2_gamma, decoder_conv2_beta,
        decoder_conv_out
    );

    DoubleConv3D<F_MAP_0, F_MAP_0, F_MAP_0>(
        decoder_conv_out,
        output_conv1_weight, output_conv1_gamma, output_conv1_beta,
        output_conv2_weight, output_conv2_gamma, output_conv2_beta,
        output_conv_out
    );

    FinalConv1x1<F_MAP_0, OUT_CHANNELS>(
        final_conv_weight,
        final_conv_bias,
        output_conv_out,
        final_conv_out
    );

    Sigmoid3D(final_conv_out, output);
}