#include "include/unet3d_reduced.h"

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

    // Array partitioning for weight arrays to enable parallel access
    #pragma HLS array_partition variable=input_conv1_weight cyclic factor=3 dim=3
    #pragma HLS array_partition variable=input_conv1_weight cyclic factor=3 dim=4
    #pragma HLS array_partition variable=input_conv1_weight cyclic factor=3 dim=5
    #pragma HLS array_partition variable=input_conv2_weight cyclic factor=3 dim=3
    #pragma HLS array_partition variable=input_conv2_weight cyclic factor=3 dim=4
    #pragma HLS array_partition variable=input_conv2_weight cyclic factor=3 dim=5
    #pragma HLS array_partition variable=encoder_conv1_weight cyclic factor=3 dim=3
    #pragma HLS array_partition variable=encoder_conv1_weight cyclic factor=3 dim=4
    #pragma HLS array_partition variable=encoder_conv1_weight cyclic factor=3 dim=5
    #pragma HLS array_partition variable=encoder_conv2_weight cyclic factor=3 dim=3
    #pragma HLS array_partition variable=encoder_conv2_weight cyclic factor=3 dim=4
    #pragma HLS array_partition variable=encoder_conv2_weight cyclic factor=3 dim=5
    #pragma HLS array_partition variable=decoder_conv1_weight cyclic factor=3 dim=3
    #pragma HLS array_partition variable=decoder_conv1_weight cyclic factor=3 dim=4
    #pragma HLS array_partition variable=decoder_conv1_weight cyclic factor=3 dim=5
    #pragma HLS array_partition variable=decoder_conv2_weight cyclic factor=3 dim=3
    #pragma HLS array_partition variable=decoder_conv2_weight cyclic factor=3 dim=4
    #pragma HLS array_partition variable=decoder_conv2_weight cyclic factor=3 dim=5
    #pragma HLS array_partition variable=output_conv1_weight cyclic factor=3 dim=3
    #pragma HLS array_partition variable=output_conv1_weight cyclic factor=3 dim=4
    #pragma HLS array_partition variable=output_conv1_weight cyclic factor=3 dim=5
    #pragma HLS array_partition variable=output_conv2_weight cyclic factor=3 dim=3
    #pragma HLS array_partition variable=output_conv2_weight cyclic factor=3 dim=4
    #pragma HLS array_partition variable=output_conv2_weight cyclic factor=3 dim=5
    #pragma HLS array_partition variable=final_conv_weight complete dim=0
    #pragma HLS array_partition variable=final_conv_bias complete

    // Array partitioning for normalization parameters
    #pragma HLS array_partition variable=input_conv1_gamma complete
    #pragma HLS array_partition variable=input_conv1_beta complete
    #pragma HLS array_partition variable=input_conv2_gamma complete
    #pragma HLS array_partition variable=input_conv2_beta complete
    #pragma HLS array_partition variable=encoder_conv1_gamma complete
    #pragma HLS array_partition variable=encoder_conv1_beta complete
    #pragma HLS array_partition variable=encoder_conv2_gamma complete
    #pragma HLS array_partition variable=encoder_conv2_beta complete
    #pragma HLS array_partition variable=decoder_conv1_gamma complete
    #pragma HLS array_partition variable=decoder_conv1_beta complete
    #pragma HLS array_partition variable=decoder_conv2_gamma complete
    #pragma HLS array_partition variable=decoder_conv2_beta complete
    #pragma HLS array_partition variable=output_conv1_gamma complete
    #pragma HLS array_partition variable=output_conv1_beta complete
    #pragma HLS array_partition variable=output_conv2_gamma complete
    #pragma HLS array_partition variable=output_conv2_beta complete

    // Intermediate buffers for dataflow

    // Input path buffers
    float input_conv_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=input_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=input_conv_out type=ram_2p impl=uram

    // Encoder path buffers
    float encoder_pool_out[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_pool_out depth=10 type=fifo
    #pragma HLS bind_storage variable=encoder_pool_out type=ram_2p impl=uram

    float encoder_conv_out[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_DEPTH][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS stream variable=encoder_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=encoder_conv_out type=ram_2p impl=uram

    // Decoder path buffers
    float decoder_upsample_out[BATCH_SIZE][F_MAP_1][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_upsample_out depth=10 type=fifo
    #pragma HLS bind_storage variable=decoder_upsample_out type=ram_2p impl=uram

    float concat_out[BATCH_SIZE][CONCAT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=concat_out depth=10 type=fifo
    #pragma HLS bind_storage variable=concat_out type=ram_2p impl=uram

    float decoder_conv_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=decoder_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=decoder_conv_out type=ram_2p impl=uram

    // Output path buffers
    float output_conv_out[BATCH_SIZE][F_MAP_0][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=output_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=output_conv_out type=ram_2p impl=uram

    float final_conv_out[BATCH_SIZE][OUTPUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS stream variable=final_conv_out depth=10 type=fifo
    #pragma HLS bind_storage variable=final_conv_out type=ram_2p impl=uram

    // UNet forward pass
    InputDoubleConv3D(input, input_conv1_weight, input_conv1_gamma, input_conv1_beta,
                      input_conv2_weight, input_conv2_gamma, input_conv2_beta, input_conv_out);

    EncoderMaxPool3D(input_conv_out, encoder_pool_out);
    EncoderDoubleConv3D(encoder_conv1_weight, encoder_conv1_gamma, encoder_conv1_beta,
                       encoder_conv2_weight, encoder_conv2_gamma, encoder_conv2_beta,
                       encoder_pool_out, encoder_conv_out);

    DecoderUpsample3D(encoder_conv_out, decoder_upsample_out);
    ConcatenateTensors(input_conv_out, decoder_upsample_out, concat_out);
    DecoderDoubleConv3D(decoder_conv1_weight, decoder_conv1_gamma, decoder_conv1_beta,
                        decoder_conv2_weight, decoder_conv2_gamma, decoder_conv2_beta,
                        concat_out, decoder_conv_out);

    OutputDoubleConv3D(decoder_conv_out, output_conv1_gamma, output_conv1_beta, output_conv1_weight,
                       output_conv2_gamma, output_conv2_beta, output_conv2_weight, output_conv_out);
    FinalConv1x1(final_conv_weight, final_conv_bias, output_conv_out, final_conv_out);
    Sigmoid3D(final_conv_out, output);
}