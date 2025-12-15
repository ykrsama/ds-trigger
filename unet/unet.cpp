#include "unet.h"

void UNet(data_t input[BATCH_SIZE][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
          data_t conv1_kernel1[F_MAP_h][INPUT_DEPTH][CONV_KERNEL][CONV_KERNEL],
          data_t conv1_kernel2[F_MAP_0][F_MAP_h][CONV_KERNEL][CONV_KERNEL],
          data_t conv2_kernel1[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL],
          data_t conv2_kernel2[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL],
          data_t conv3_kernel1[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL],
          data_t conv3_kernel2[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL],
          data_t final_kernel[OUT_CHANNELS][F_MAP_0][1][1][1],
          data_t final_bias[OUT_CHANNELS],
          data_t output[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]) {

    #pragma HLS interface mode=m_axi port=input offset=slave bundle=gmem0
    #pragma HLS interface mode=m_axi port=conv1_kernel1 offset=slave bundle=gmem1
    #pragma HLS interface mode=m_axi port=conv1_kernel2 offset=slave bundle=gmem2
    #pragma HLS interface mode=m_axi port=conv2_kernel1 offset=slave bundle=gmem3
    #pragma HLS interface mode=m_axi port=conv2_kernel2 offset=slave bundle=gmem4
    #pragma HLS interface mode=m_axi port=conv3_kernel1 offset=slave bundle=gmem5
    #pragma HLS interface mode=m_axi port=conv3_kernel2 offset=slave bundle=gmem6
    #pragma HLS interface mode=m_axi port=final_kernel offset=slave bundle=gmem7
    #pragma HLS interface mode=m_axi port=final_bias offset=slave bundle=gmem8
    #pragma HLS interface mode=m_axi port=output offset=slave bundle=gmem9
    #pragma HLS interface mode=s_axilite port=return

    // Intermediate buffers
    data_t skip_connection[BATCH_SIZE][F_MAP_0][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=skip_connection type=ram_t2p impl=bram
    #pragma HLS array_partition variable=skip_connection cyclic factor=F_MAP_0 dim=2

    data_t encoder_output[BATCH_SIZE][F_MAP_0][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=encoder_output type=ram_t2p impl=bram
    #pragma HLS array_partition variable=encoder_output cyclic factor=F_MAP_0 dim=2

    data_t pooled_output[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS bind_storage variable=pooled_output type=ram_t2p impl=bram
    #pragma HLS array_partition variable=pooled_output cyclic factor=F_MAP_0 dim=2

    data_t encoder_conv_output[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH];
    #pragma HLS bind_storage variable=encoder_conv_output type=ram_t2p impl=bram
    #pragma HLS array_partition variable=encoder_conv_output cyclic factor=F_MAP_1 dim=2

    data_t upsampled_output[BATCH_SIZE][F_MAP_1][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=upsampled_output type=ram_t2p impl=bram
    #pragma HLS array_partition variable=upsampled_output cyclic factor=F_MAP_1 dim=2

    data_t concat_output[BATCH_SIZE][CONCAT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=concat_output type=ram_t2p impl=bram
    #pragma HLS array_partition variable=concat_output cyclic factor=CONCAT_CHANNELS dim=2

    data_t decoder_output[BATCH_SIZE][F_MAP_0][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=decoder_output type=ram_t2p impl=bram
    #pragma HLS array_partition variable=decoder_output cyclic factor=F_MAP_0 dim=2

    data_t final_conv_output[BATCH_SIZE][OUT_CHANNELS * INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=final_conv_output type=ram_t2p impl=bram
    #pragma HLS array_partition variable=final_conv_output cyclic factor=(OUT_CHANNELS * INPUT_DEPTH) dim=2

    // Input: DoubleConv2D2Head chw 11 x 43 x 43 -> F_MAP_h x 43 x 43 -> F_MAP_0 x 43 x 43
    DoubleConv2D2Head<INPUT_DEPTH, F_MAP_h, F_MAP_0, INPUT_HEIGHT, INPUT_WIDTH>(
        input, conv1_kernel1, conv1_kernel2, skip_connection, encoder_output);

    // Encode: MaxPool2D chw F_MAP_0 x 43 x 43 -> F_MAP_0 x POOL_OUTPUT_HEIGHT x POOL_OUTPUT_WIDTH
    MaxPool2D<F_MAP_0>(encoder_output, pooled_output);

    // Encode: DoubleConv2D chw F_MAP_0 x POOL_OUTPUT_HEIGHT x POOL_OUTPUT_WIDTH -> F_MAP_1 x POOL_OUTPUT_HEIGHT x POOL_OUTPUT_WIDTH
    DoubleConv2D<F_MAP_0, F_MAP_0, F_MAP_1, POOL_OUTPUT_HEIGHT, POOL_OUTPUT_WIDTH>(
        pooled_output, conv2_kernel1, conv2_kernel2, encoder_conv_output);

    // Decode: Upsample2D chw F_MAP_1 x POOL_OUTPUT_HEIGHT x POOL_OUTPUT_WIDTH -> F_MAP_1 x 43 x 43
    Upsample2D<F_MAP_1>(
        encoder_conv_output, upsampled_output);

    // Decode: Concat2D chw (F_MAP_0 x 43 x 43 + F_MAP_1 x 43 x 43) -> CONCAT_CHANNELS x 43 x 43
    Concat2D<F_MAP_0, F_MAP_1, CONCAT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH>(
        skip_connection, upsampled_output, concat_output);

    // Decode: DoubleConv2D chw CONCAT_CHANNELS x 43 x 43 -> F_MAP_0 x 43 x 43
    DoubleConv2D<CONCAT_CHANNELS, F_MAP_0, F_MAP_0, INPUT_HEIGHT, INPUT_WIDTH>(
        concat_output, conv3_kernel1, conv3_kernel2, decoder_output);

    // Output: finalconv chw F_MAP_0 x 43 x 43 -> 55 x 43 x 43 (OUT_CHANNELS * INPUT_DEPTH)
    FinalConv2D<F_MAP_0, (OUT_CHANNELS * INPUT_DEPTH), INPUT_HEIGHT, INPUT_WIDTH>(
        final_kernel, final_bias, decoder_output, final_conv_output);

    // Reshape: chw 55 x 43 x 43 -> cdhw OUT_CHANNELS x 11 x 43 x 43
    ReshapeOutput<(OUT_CHANNELS * INPUT_DEPTH), OUT_CHANNELS, INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH>(
        final_conv_output, output);
}

// Helper function implementations based on templates from model.cpp

template<int T_IN_CHANNELS,
         int T_MID_CHANNELS,
         int T_OUT_CHANNELS,
         int T_INPUT_HEIGHT,
         int T_INPUT_WIDTH>
void DoubleConv2D(data_t input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                  data_t kernel1[T_MID_CHANNELS][T_IN_CHANNELS][CONV_KERNEL][CONV_KERNEL],
                  data_t kernel2[T_OUT_CHANNELS][T_MID_CHANNELS][CONV_KERNEL][CONV_KERNEL],
                  data_t output[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH]) {

    data_t conv1_out[BATCH_SIZE][T_MID_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH];
    #pragma HLS bind_storage variable=conv1_out type=ram_t2p impl=bram

    Conv2D<T_IN_CHANNELS, T_MID_CHANNELS, T_INPUT_HEIGHT, T_INPUT_WIDTH>(kernel1, input, conv1_out);
    Conv2D<T_MID_CHANNELS, T_OUT_CHANNELS, T_INPUT_HEIGHT, T_INPUT_WIDTH>(kernel2, conv1_out, output);
}

template<int T_CHANNELS1, int T_CHANNELS2, int T_CONCAT_CHANNELS, int T_INPUT_HEIGHT, int T_INPUT_WIDTH>
void Concat2D(data_t tensor1[BATCH_SIZE][T_CHANNELS1][T_INPUT_HEIGHT][T_INPUT_WIDTH],
              data_t tensor2[BATCH_SIZE][T_CHANNELS2][T_INPUT_HEIGHT][T_INPUT_WIDTH],
              data_t output[BATCH_SIZE][T_CONCAT_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH]) {
    #pragma HLS array_partition variable=tensor1 cyclic factor=T_CHANNELS1 dim=2
    #pragma HLS array_partition variable=tensor2 cyclic factor=T_CHANNELS2 dim=2
    #pragma HLS array_partition variable=output cyclic factor=T_CONCAT_CHANNELS dim=2

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int height = 0; height < T_INPUT_HEIGHT; height++) {
            for (int width = 0; width < T_INPUT_WIDTH; width++) {
                #pragma HLS pipeline II=1
                // Process tensor1 channels in parallel
                for (int ch = 0; ch < T_CHANNELS1; ch++) {
                    #pragma HLS unroll
                    output[batch][ch][height][width] = tensor1[batch][ch][height][width];
                }

                // Process tensor2 channels in parallel
                for (int ch = 0; ch < T_CHANNELS2; ch++) {
                    #pragma HLS unroll
                    output[batch][T_CHANNELS1 + ch][height][width] = tensor2[batch][ch][height][width];
                }
            }
        }
    }
}

template<int T_IN_CHANNELS, int T_OUT_CHANNELS, int T_INPUT_HEIGHT, int T_INPUT_WIDTH>
void FinalConv2D(data_t kernel[T_OUT_CHANNELS][T_IN_CHANNELS][1][1][1],
                 data_t bias[T_OUT_CHANNELS],
                 data_t input[BATCH_SIZE][T_IN_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH],
                 data_t output[BATCH_SIZE][T_OUT_CHANNELS][T_INPUT_HEIGHT][T_INPUT_WIDTH]) {
    #pragma HLS array_partition variable=kernel complete dim=2
    #pragma HLS array_partition variable=bias complete dim=1
    #pragma HLS array_partition variable=input complete dim=2
    #pragma HLS array_partition variable=output complete dim=2

    ConvBatch:
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        ConvHeight:
        for (int height = 0; height < T_INPUT_HEIGHT; height++) {
            ConvWidth:
            for (int width = 0; width < T_INPUT_WIDTH; width++) {
                #pragma HLS pipeline II=1
                data_t accum[T_OUT_CHANNELS];
                #pragma HLS bind_storage variable=accum type=ram_t2p impl=bram

                ConvMAC:
                for (int out_ch = 0; out_ch < T_OUT_CHANNELS; out_ch++) {
                    #pragma HLS unroll
                    accum[out_ch] = bias[out_ch];

                    for (int in_ch = 0; in_ch < T_IN_CHANNELS; in_ch++) {
                        accum[out_ch] += input[batch][in_ch][height][width] *
                                         kernel[out_ch][in_ch][0][0][0];
                    }

                    output[batch][out_ch][height][width] = accum[out_ch];
                }
            }
        }
    }
}

template<int T_IN_CHANNELS, int T_OUT_CHANNELS, int T_DEPTH, int T_HEIGHT, int T_WIDTH>
void ReshapeOutput(data_t input[BATCH_SIZE][T_IN_CHANNELS][T_HEIGHT][T_WIDTH],
                   data_t output[BATCH_SIZE][T_OUT_CHANNELS][T_DEPTH][T_HEIGHT][T_WIDTH]) {
    #pragma HLS array_partition variable=input complete dim=2
    #pragma HLS array_partition variable=output complete dim=2

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int height = 0; height < T_HEIGHT; height++) {
            for (int width = 0; width < T_WIDTH; width++) {
                #pragma HLS pipeline II=1
                for (int flat_ch = 0; flat_ch < T_IN_CHANNELS; flat_ch++) {
                    #pragma HLS unroll
                    int out_ch = flat_ch % T_OUT_CHANNELS;
                    int out_depth = flat_ch / T_OUT_CHANNELS;

                    if (out_depth < T_DEPTH) {
                        output[batch][out_ch][out_depth][height][width] = input[batch][flat_ch][height][width];
                    }
                }
            }
        }
    }
}


// Template instantiations for DoubleConv2D
template void DoubleConv2D<F_MAP_0, F_MAP_0, F_MAP_1, POOL_OUTPUT_HEIGHT, POOL_OUTPUT_WIDTH>(
    data_t[BATCH_SIZE][F_MAP_0][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH],
    data_t[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL],
    data_t[F_MAP_1][F_MAP_0][CONV_KERNEL][CONV_KERNEL],
    data_t[BATCH_SIZE][F_MAP_1][POOL_OUTPUT_HEIGHT][POOL_OUTPUT_WIDTH]);

template void DoubleConv2D<CONCAT_CHANNELS, F_MAP_0, F_MAP_0, INPUT_HEIGHT, INPUT_WIDTH>(
    data_t[BATCH_SIZE][CONCAT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH],
    data_t[F_MAP_0][CONCAT_CHANNELS][CONV_KERNEL][CONV_KERNEL],
    data_t[F_MAP_0][F_MAP_0][CONV_KERNEL][CONV_KERNEL],
    data_t[BATCH_SIZE][F_MAP_0][INPUT_HEIGHT][INPUT_WIDTH]);


// Template instantiations for ReshapeOutput (our implementation)
template void ReshapeOutput<(OUT_CHANNELS * INPUT_DEPTH), OUT_CHANNELS, INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH>(
    data_t[BATCH_SIZE][OUT_CHANNELS * INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    data_t[BATCH_SIZE][OUT_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);

// Template instantiations for Concat2D (our implementation)
template void Concat2D<F_MAP_0, F_MAP_1, CONCAT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH>(
    data_t[BATCH_SIZE][F_MAP_0][INPUT_HEIGHT][INPUT_WIDTH],
    data_t[BATCH_SIZE][F_MAP_1][INPUT_HEIGHT][INPUT_WIDTH],
    data_t[BATCH_SIZE][CONCAT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH]);

// Template instantiations for FinalConv2D (our implementation)
template void FinalConv2D<F_MAP_0, (OUT_CHANNELS * INPUT_DEPTH), INPUT_HEIGHT, INPUT_WIDTH>(
    data_t[OUT_CHANNELS * INPUT_DEPTH][F_MAP_0][1][1][1],
    data_t[OUT_CHANNELS * INPUT_DEPTH],
    data_t[BATCH_SIZE][F_MAP_0][INPUT_HEIGHT][INPUT_WIDTH],
    data_t[BATCH_SIZE][OUT_CHANNELS * INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH]);
