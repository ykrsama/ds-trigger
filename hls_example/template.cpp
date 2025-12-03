//------------------------模板, 将对应的函数复制，一键替换对应的变量名即可----------------------------------

// BATCH_SIZE - 批次大小
// IN_CHANNELS - 输入通道数
// OUT_CHANNELS - 输出通道数  
// INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH - 输入尺寸
// KERNEL - 卷积核尺寸
// STRIDE - 步长
// PADDING - 填充大小
// OUTPUT_DEPTH = (INPUT_DEPTH + 2 * PADDING - KERNEL) / STRIDE + 1;
// OUTPUT_HEIGHT = (INPUT_HEIGHT + 2 * PADDING - KERNEL) / STRIDE + 1;
// OUTPUT_WIDTH = (INPUT_WIDTH + 2 * PADDING - KERNEL) / STRIDE + 1;

void Conv3d(
  float kernel[OUT_CHANNELS][IN_CHANNELS][KERNEL][KERNEL][KERNEL],
  float input[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
  float output[BATCH_SIZE][OUT_CHANNELS][OUTPUT_DEPTH][OUTPUT_HEIGHT][OUTPUT_WIDTH]
) {
  #pragma HLS array_partition variable=kernel cyclic factor=IN_CHANNELS dim=2
  #pragma HLS array_partition variable=kernel cyclic factor=KERNEL dim=3
  #pragma HLS array_partition variable=kernel cyclic factor=KERNEL dim=4
  #pragma HLS array_partition variable=kernel cyclic factor=KERNEL dim=5

  // 填充输入
  float padded_input[BATCH_SIZE][IN_CHANNELS][PADDED_DEPTH][PADDED_HEIGHT][PADDED_WIDTH];
  #pragma HLS stream variable=padded_input depth=10 type=fifo
  #pragma HLS bind_storage variable=padded_input type=ram_t2p impl=bram

  // 填充操作
  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int depth = 0; depth < PADDED_DEPTH; depth++) {
      for (int height = 0; height < PADDED_HEIGHT; height++) {
        for (int width = 0; width < PADDED_WIDTH; width++) {
          for (int in_ch = 0; in_ch < IN_CHANNELS; in_ch++) {
            float pad_value = (float)0.000000;
            int orig_depth = depth - PADDING;
            int orig_height = height - PADDING;
            int orig_width = width - PADDING;
            
            if (orig_depth >= 0 && orig_depth < INPUT_DEPTH &&
                orig_height >= 0 && orig_height < INPUT_HEIGHT &&
                orig_width >= 0 && orig_width < INPUT_WIDTH) {
              pad_value = input[batch][in_ch][orig_depth][orig_height][orig_width];
            }
            padded_input[batch][in_ch][depth][height][width] = pad_value;
          }
        }
      }
    }
  }

  // 卷积输出
  float [BATCH_SIZE][OUT_CHANNELS][OUTPUT_DEPTH][OUTPUT_HEIGHT][OUTPUT_WIDTH];
  #pragma HLS stream variable=conv_out depth=10 type=fifo
  #pragma HLS bind_storage variable=conv_out type=ram_t2p impl=bram

  // 缓冲器定义
  float cube_buffer[BATCH_SIZE][IN_CHANNELS][KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
  // #pragma HLS array_partition variable=cube_buffer cyclic factor=KERNEL dim=3
  #pragma HLS bind_storage variable=cube_buffer type=ram_2p impl=lutram

  float line_buffer[BATCH_SIZE][IN_CHANNELS][KERNEL][KERNEL][PADDED_WIDTH];
  // #pragma HLS array_partition variable=line_buffer cyclic factor=KERNEL dim=3
  // #pragma HLS array_partition variable=line_buffer cyclic factor=KERNEL dim=4
  #pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

  float window_buffer[BATCH_SIZE][IN_CHANNELS][KERNEL][KERNEL][KERNEL];
  #pragma HLS array_partition variable=window_buffer cyclic factor=IN_CHANNELS dim=2
  #pragma HLS array_partition variable=window_buffer cyclic factor=KERNEL dim=3
  #pragma HLS array_partition variable=window_buffer cyclic factor=KERNEL dim=4
  #pragma HLS array_partition variable=window_buffer cyclic factor=KERNEL dim=5
  #pragma HLS bind_storage variable=window_buffer type=ram_2p impl=lutram


  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int depth = 0; depth < PADDED_DEPTH; depth++) {
      for (int height = 0; height < PADDED_HEIGHT; height++) {
        for (int width = 0; width < PADDED_WIDTH; width++) {
          
          // 更新cube buffer
          for (int in_ch = 0; in_ch < IN_CHANNELS; in_ch++) {
            for (int kd = 0; kd < KERNEL - 1; kd++) {
              cube_buffer[batch][in_ch][kd][height][width] = 
                cube_buffer[batch][in_ch][kd + 1][height][width];
            }
            cube_buffer[batch][in_ch][KERNEL - 1][height][width] = 
              padded_input[batch][in_ch][depth][height][width];
          }
          
          if (depth >= KERNEL - 1) {
            // 更新line buffer
            for (int in_ch = 0; in_ch < IN_CHANNELS; in_ch++) {
              for (int kd = 0; kd < KERNEL; kd++) {
                for (int kh = 0; kh < KERNEL - 1; kh++) {
                  line_buffer[batch][in_ch][kd][kh][width] = 
                    line_buffer[batch][in_ch][kd][kh + 1][width];
                }
                line_buffer[batch][in_ch][kd][KERNEL - 1][width] = 
                  cube_buffer[batch][in_ch][kd][height][width];
              }
            }
            
            if (height >= KERNEL - 1) {
              // 更新window buffer
              for (int in_ch = 0; in_ch < IN_CHANNELS; in_ch++) {
                for (int kd = 0; kd < KERNEL; kd++) {
                  for (int kh = 0; kh < KERNEL; kh++) {
                    for (int kw = 0; kw < KERNEL - 1; kw++) {
                      window_buffer[batch][in_ch][kd][kh][kw] = 
                        window_buffer[batch][in_ch][kd][kh][kw + 1];
                    }
                    window_buffer[batch][in_ch][kd][kh][KERNEL - 1] = 
                      line_buffer[batch][in_ch][kd][kh][width];
                  }
                }
              }
              
              // 卷积计算
              if (width >= KERNEL - 1 && 
                  (depth) % STRIDE == 0 &&
                  (height) % STRIDE == 0 &&
                  (width) % STRIDE == 0) {
                
                float accum[OUT_CHANNELS];
                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram
                
                for (int out_ch = 0; out_ch < OUT_CHANNELS; out_ch++) {
                  #pragma HLS pipeline II=1
                  accum[out_ch] = (float)0.000000;
                  
                  for (int in_ch = 0; in_ch < IN_CHANNELS; in_ch++) {
                    for (int kd = 0; kd < KERNEL; kd++) {
                      for (int kh = 0; kh < KERNEL; kh++) {
                        for (int kw = 0; kw < KERNEL; kw++) {
                          float window_val = window_buffer[batch][in_ch][kd][kh][kw];
                          float kernel_val = kernel[out_ch][in_ch][kd][kh][kw];
                          accum[out_ch] += window_val * kernel_val;
                        }
                      }
                    }
                  }
                  
                  int out_depth = (depth - KERNEL) / STRIDE + 1;
                  int out_height = (height - KERNEL) / STRIDE + 1;
                  int out_width = (width - KERNEL) / STRIDE + 1;
                  
                  // relu(output)
                  float output_value = accum[out_ch];
                  bool is_positive = output_value > (float)0.000000;	
                  float relu_output = is_positive ? output_value : (float)0.000000;
                  output[batch][out_ch][out_depth][out_height][out_width] = relu_output;
                  
                }
              }
            }
          }
        }
      }
    }
  }
}

// BATCH_SIZE - 批次大小
// IN_CHANNELS - 输入通道数
// INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH - 输入尺寸
// KERNEL - 池化核尺寸
// STRIDE - 步长

// OUTPUT_DEPTH = (INPUT_DEPTH - KERNEL) / STRIDE + 1;
// OUTPUT_HEIGHT = (INPUT_HEIGHT - KERNEL) / STRIDE + 1;
// OUTPUT_WIDTH = (INPUT_WIDTH - KERNEL) / STRIDE + 1;

void MaxPool3d(
  float input[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
  float output[BATCH_SIZE][IN_CHANNELS][OUTPUT_DEPTH][OUTPUT_HEIGHT][OUTPUT_WIDTH]
) {
  #pragma HLS dataflow


// 缓冲器定义
  float cube_buffer[BATCH_SIZE][IN_CHANNELS][KERNEL][PADDED_HEIGHT][PADDED_WIDTH];
  // #pragma HLS array_partition variable=cube_buffer cyclic factor=KERNEL dim=3
  #pragma HLS bind_storage variable=cube_buffer type=ram_2p impl=lutram

  float line_buffer[BATCH_SIZE][IN_CHANNELS][KERNEL][KERNEL][PADDED_WIDTH];
  // #pragma HLS array_partition variable=line_buffer cyclic factor=KERNEL dim=3
  // #pragma HLS array_partition variable=line_buffer cyclic factor=KERNEL dim=4
  #pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

  float window_buffer[BATCH_SIZE][IN_CHANNELS][KERNEL][KERNEL][KERNEL];
  #pragma HLS array_partition variable=window_buffer cyclic factor=IN_CHANNELS dim=2
  #pragma HLS array_partition variable=window_buffer cyclic factor=KERNEL dim=3
  #pragma HLS array_partition variable=window_buffer cyclic factor=KERNEL dim=4
  #pragma HLS array_partition variable=window_buffer cyclic factor=KERNEL dim=5
  #pragma HLS bind_storage variable=window_buffer type=ram_2p impl=lutram


  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int depth = 0; depth < PADDED_DEPTH; depth++) {
      for (int height = 0; height < PADDED_HEIGHT; height++) {
        for (int width = 0; width < PADDED_WIDTH; width++) {
          
          // 更新cube buffer
          for (int in_ch = 0; in_ch < IN_CHANNELS; in_ch++) {
            for (int kd = 0; kd < KERNEL - 1; kd++) {
              cube_buffer[batch][in_ch][kd][height][width] = 
                cube_buffer[batch][in_ch][kd + 1][height][width];
            }
            cube_buffer[batch][in_ch][KERNEL - 1][height][width] = 
              input[batch][in_ch][depth][height][width];
          }
          
          if (depth >= KERNEL - 1) {
            // 更新line buffer
            for (int in_ch = 0; in_ch < IN_CHANNELS; in_ch++) {
              for (int kd = 0; kd < KERNEL; kd++) {
                for (int kh = 0; kh < KERNEL - 1; kh++) {
                  line_buffer[batch][in_ch][kd][kh][width] = 
                    line_buffer[batch][in_ch][kd][kh + 1][width];
                }
                line_buffer[batch][in_ch][kd][KERNEL - 1][width] = 
                  cube_buffer[batch][in_ch][kd][height][width];
              }
            }
            
            if (height >= KERNEL - 1) {
              // 更新window buffer
              for (int in_ch = 0; in_ch < IN_CHANNELS; in_ch++) {
                for (int kd = 0; kd < KERNEL; kd++) {
                  for (int kh = 0; kh < KERNEL; kh++) {
                    for (int kw = 0; kw < KERNEL - 1; kw++) {
                      window_buffer[batch][in_ch][kd][kh][kw] = 
                        window_buffer[batch][in_ch][kd][kh][kw + 1];
                    }
                    window_buffer[batch][in_ch][kd][kh][KERNEL - 1] = 
                      line_buffer[batch][in_ch][kd][kh][width];
                  }
                }
              }
              
              // 池化计算
              if (width >= KERNEL - 1 && 
                  (depth) % STRIDE == 0 &&
                  (height) % STRIDE == 0 &&
                  (width) % STRIDE == 0) {
                
                float accum[OUT_CHANNELS];
                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram
                
                for (int out_ch = 0; out_ch < OUT_CHANNELS; out_ch++) {
                  #pragma HLS pipeline II=1
                  accum[out_ch] = (float)-INFINITY;;
                  
                  for (int in_ch = 0; in_ch < IN_CHANNELS; in_ch++) {
                    for (int kd = 0; kd < KERNEL; kd++) {
                      for (int kh = 0; kh < KERNEL; kh++) {
                        for (int kw = 0; kw < KERNEL; kw++) {
                          float window_val = window_buffer[batch][in_ch][kd][kh][kw];
                          float accum_temp = accum[out_ch];
                          accum[out_ch] = max(accum_temp, window_val);
                        }
                      }
                    }
                  }
                  
                  int out_depth = (depth - KERNEL) / STRIDE + 1;
                  int out_height = (height - KERNEL) / STRIDE + 1;
                  int out_width = (width - KERNEL) / STRIDE + 1;
                  
                  float output_value = accum[out_ch];
                  output[batch][out_ch][out_depth][out_height][out_width] = output_value;
                  
                }
              }
            }
          }
        }
      }
    }
  }
}



// BATCH_SIZE - 批次大小
// IN_CHANNELS - 输入通道数
// INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH - 输入尺寸
// NUM_GROUPS - 分组数

void GroupNorm3d(
    float input_data[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float gamma[IN_CHANNELS],
    float beta[IN_CHANNELS],
    float output_data[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH],
    float EPSILON, int NUM_GROUPS
) {
    // 计算每组通道数
    int CHANNELS_PER_GROUP = IN_CHANNELS / NUM_GROUPS;
    float N = (float)(INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH * CHANNELS_PER_GROUP);

    // --- 1. 定义片上缓存 ---
    float gn_buffer[BATCH_SIZE][IN_CHANNELS][INPUT_DEPTH][INPUT_HEIGHT][INPUT_WIDTH];
    #pragma HLS bind_storage variable=gn_buffer type=ram_2p impl=uram

    // 定义统计量
    float group_sum[NUM_GROUPS];
    float group_sq_sum[NUM_GROUPS];
    #pragma HLS array_partition variable=group_sum complete
    #pragma HLS array_partition variable=group_sq_sum complete

    for (int g = 0; g < NUM_GROUPS; g++) {
        #pragma HLS unroll
        group_sum[g] = (float)0.000000;
        group_sq_sum[g] = (float)0.000000;
    }


    for (int batch = 0; batch < BATCH_SIZE; batch++) {
      for (int ch = 0; ch < IN_CHANNELS; ch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                      // 计算当前通道属于哪个Group
                      int group_idx = ch / CHANNELS_PER_GROUP;

                      // 读取输入数据
                      float value = input_data[batch][ch][depth][height][width];

                      // 写入缓存
                      gn_buffer[batch][ch][depth][height][width] = value;

                      // 累加统计量
                      group_sum[group_idx] += value;
                      group_sq_sum[group_idx] += (value * value);
                    }
                }
            }
        }
    }


    float mean[NUM_GROUPS];
    float inv_std[NUM_GROUPS];
    #pragma HLS array_partition variable=mean complete
    #pragma HLS array_partition variable=inv_std complete


    for (int g = 0; g < NUM_GROUPS; g++) {
        #pragma HLS unroll
        float mu = group_sum[g] / N;
        float variance = (group_sq_sum[g] / N) - (mu * mu);
        float sigma = sqrt(variance + EPSILON);
        
        mean[g] = mu;
        inv_std[g] = float(1) / sigma;
    }


    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int depth = 0; depth < INPUT_DEPTH; depth++) {
            for (int height = 0; height < INPUT_HEIGHT; height++) {
                for (int width = 0; width < INPUT_WIDTH; width++) {
                    #pragma HLS pipeline II=1
                    for (int ch = 0; ch < IN_CHANNELS; ch++) {
                        // 计算当前通道属于哪个Group
                        int group_idx = ch / CHANNELS_PER_GROUP;

                        // 从缓存读取数据
                        float value = gn_buffer[batch][ch][depth][height][width];
                        
                        // 获取对应的Gamma和Beta参数
                        float gamma_param = gamma[ch];
                        float beta_param = beta[ch];

                        // 获取对应的统计量
                        float group_mean = mean[group_idx];
                        float group_inv_std = inv_std[group_idx];

                        // 标准化: (x - mean) * inv_std
                        float normalized_value = (value - group_mean) * group_inv_std;

                        // 仿射变换: normalized * gamma + beta
                        float output_value = normalized_value * gamma_param + beta_param;

                        // 输出
                        // bool is_positive = output_value > (float)0.000000;	
                        // float relu_output = is_positive ? output_value : (float)0.000000;
                        output_data[batch][ch][depth][height][width] = output_value;

                        
                    }
                }
            }
        }
    }
}




