//-------------------------------------  crg实现例子  ------------------------------

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

using namespace std;


void Conv3d(
  float kernel[64][1][3][3][3],
  float input[1][1][43][43][11],
  float output[1][64][43][43][11]
) {
  #pragma HLS dataflow

  #pragma HLS array_partition variable=kernel cyclic factor=1 dim=2
  #pragma HLS array_partition variable=kernel cyclic factor=3 dim=3
  #pragma HLS array_partition variable=kernel cyclic factor=3 dim=4
  #pragma HLS array_partition variable=kernel cyclic factor=3 dim=5
  
  // 填充后尺寸， 这里注意也要计算后替换！！！
  // int 45 = 43 + 2 * 1;
  // int 45 = 43 + 2 * 1;
  // int 13 = 11 + 2 * 1;

  // 填充输入
  float padded_input[1][1][45][45][13];
  #pragma HLS stream variable=padded_input depth=10 type=fifo
  #pragma HLS bind_storage variable=padded_input type=ram_t2p impl=bram

  // 填充操作
  for (int batch = 0; batch < 1; batch++) {
    for (int depth = 0; depth < 45; depth++) {
      for (int height = 0; height < 45; height++) {
        for (int width = 0; width < 13; width++) {
          for (int in_ch = 0; in_ch < 1; in_ch++) {
            float pad_value = (float)0.000000;
            int orig_depth = depth - 1;
            int orig_height = height - 1;
            int orig_width = width - 1;
            
            if (orig_depth >= 0 && orig_depth < 43 &&
                orig_height >= 0 && orig_height < 43 &&
                orig_width >= 0 && orig_width < 11) {
              pad_value = input[batch][in_ch][orig_depth][orig_height][orig_width];
            }
            padded_input[batch][in_ch][depth][height][width] = pad_value;
          }
        }
      }
    }
  }


  // 缓冲器定义
  float cube_buffer[1][1][3][45][13];
  // #pragma HLS array_partition variable=cube_buffer cyclic factor=3 dim=3
  #pragma HLS bind_storage variable=cube_buffer type=ram_2p impl=lutram

  float line_buffer[1][1][3][3][13];
  // #pragma HLS array_partition variable=line_buffer cyclic factor=3 dim=3
  // #pragma HLS array_partition variable=line_buffer cyclic factor=3 dim=4
  #pragma HLS bind_storage variable=line_buffer type=ram_2p impl=lutram

  float window_buffer[1][1][3][3][3];
  #pragma HLS array_partition variable=window_buffer cyclic factor=1 dim=2
  #pragma HLS array_partition variable=window_buffer cyclic factor=3 dim=3
  #pragma HLS array_partition variable=window_buffer cyclic factor=3 dim=4
  #pragma HLS array_partition variable=window_buffer cyclic factor=3 dim=5
  #pragma HLS bind_storage variable=window_buffer type=ram_2p impl=lutram


  for (int batch = 0; batch < 1; batch++) {
    for (int depth = 0; depth < 45; depth++) {
      for (int height = 0; height < 45; height++) {
        for (int width = 0; width < 13; width++) {
          
          // 更新cube buffer
          for (int in_ch = 0; in_ch < 1; in_ch++) {
            for (int kd = 0; kd < 3 - 1; kd++) {
              cube_buffer[batch][in_ch][kd][height][width] = 
                cube_buffer[batch][in_ch][kd + 1][height][width];
            }
            cube_buffer[batch][in_ch][3 - 1][height][width] = 
              padded_input[batch][in_ch][depth][height][width];
          }
          
          if (depth >= 3 - 1) {
            // 更新line buffer
            for (int in_ch = 0; in_ch < 1; in_ch++) {
              for (int kd = 0; kd < 3; kd++) {
                for (int kh = 0; kh < 3 - 1; kh++) {
                  line_buffer[batch][in_ch][kd][kh][width] = 
                    line_buffer[batch][in_ch][kd][kh + 1][width];
                }
                line_buffer[batch][in_ch][kd][3 - 1][width] = 
                  cube_buffer[batch][in_ch][kd][height][width];
              }
            }
            
            if (height >= 3 - 1) {
              // 更新window buffer
              for (int in_ch = 0; in_ch < 1; in_ch++) {
                for (int kd = 0; kd < 3; kd++) {
                  for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3 - 1; kw++) {
                      window_buffer[batch][in_ch][kd][kh][kw] = 
                        window_buffer[batch][in_ch][kd][kh][kw + 1];
                    }
                    window_buffer[batch][in_ch][kd][kh][3 - 1] = 
                      line_buffer[batch][in_ch][kd][kh][width];
                  }
                }
              }
              
              // 卷积计算
              if (width >= 3 - 1 && 
                  (depth) % 1 == 0 &&
                  (height) % 1 == 0 &&
                  (width) % 1 == 0) {
                
                float accum[64];
                #pragma HLS bind_storage variable=accum type=ram_2p impl=bram
                
                for (int out_ch = 0; out_ch < 64; out_ch++) {
                  #pragma HLS pipeline II=1
                  accum[out_ch] = (float)0.000000;
                  
                  for (int in_ch = 0; in_ch < 1; in_ch++) {
                    for (int kd = 0; kd < 3; kd++) {
                      for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                          float window_val = window_buffer[batch][in_ch][kd][kh][kw];
                          float kernel_val = kernel[out_ch][in_ch][kd][kh][kw];
                          accum[out_ch] += window_val * kernel_val;
                        }
                      }
                    }
                  }
                  
                  int out_depth = (depth - 3) / 1 + 1;
                  int out_height = (height - 3) / 1 + 1;
                  int out_width = (width - 3) / 1 + 1;
                  
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

void GroupNorm3d(
    float input_data[1][64][43][43][11],
    float gamma[64],
    float beta[64],
    float output_data[1][64][43][43][11],
    float EPSILON// int 8， num_groups也要替换！！！
) {
    // 计算每组通道数
    int CHANNELS_PER_GROUP = 64 / 8;
    float N = (float)(43 * 43 * 11 * CHANNELS_PER_GROUP);

    // --- 1. 定义片上缓存 ---
    float gn_buffer[1][64][43][43][11];
    #pragma HLS bind_storage variable=gn_buffer type=ram_2p impl=uram

    // 定义统计量
    float group_sum[8];
    float group_sq_sum[8];
    #pragma HLS array_partition variable=group_sum complete
    #pragma HLS array_partition variable=group_sq_sum complete

    for (int g = 0; g < 8; g++) {
        #pragma HLS unroll
        group_sum[g] = (float)0.000000;
        group_sq_sum[g] = (float)0.000000;
    }


    for (int batch = 0; batch < 1; batch++) {
      
        for (int depth = 0; depth < 43; depth++) {
            for (int height = 0; height < 43; height++) {
                for (int width = 0; width < 11; width++) {
                  for (int ch = 0; ch < 64; ch++) {
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


    float mean[8];
    float inv_std[8];
    #pragma HLS array_partition variable=mean complete
    #pragma HLS array_partition variable=inv_std complete


    for (int g = 0; g < 8; g++) {
        #pragma HLS unroll
        float mu = group_sum[g] / N;
        float variance = (group_sq_sum[g] / N) - (mu * mu);
        float sigma = sqrt(variance + EPSILON);
        
        mean[g] = mu;
        inv_std[g] = float(1) / sigma;
    }


    for (int batch = 0; batch < 1; batch++) {
        for (int depth = 0; depth < 43; depth++) {
            for (int height = 0; height < 43; height++) {
                for (int width = 0; width < 11; width++) {
                    #pragma HLS pipeline II=1
                    for (int ch = 0; ch < 64; ch++) {
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





  // here is example: order=crg, conv+relu+groupnorm
void doubleconv( //top function，这里可以传入参数，从cpu->device off-chip mem -> on-chip mem
  float input[1][1][43][43][11],
  float kernel[64][1][3][3][3],
  float gamma[64],
  float beta[64],
  float epsilon, int num_groups,
  float output[1][64][43][43][11]
) {	// L187
  
  #pragma HLS interface s_axilite port=return 
  #pragma HLS interface bram port=input
  #pragma HLS bind_storage variable=input type=ram_t2p impl=bram
  #pragma HLS interface bram port=kernel
  #pragma HLS bind_storage variable=kernel type=ram_t2p impl=bram
  #pragma HLS interface bram port=gamma
  #pragma HLS bind_storage variable=gamma type=ram_t2p impl=bram
  #pragma HLS interface bram port=beta
  #pragma HLS bind_storage variable=beta type=ram_t2p impl=bram
  #pragma HLS interface bram port=output
  #pragma HLS bind_storage variable=output type=ram_t2p impl=bram

  #pragma HLS dataflow

  float conv_out[1][64][43][43][11];	//这里用标注为fifo的conv_out连接不同的算子，使得整体fifo,改为stream<float>也可
  #pragma HLS stream variable=conv_out depth=10 type=fifo
  Conv3d(kernel, input, conv_out);
  GroupNorm3d(conv_out, gamma, beta, output, epsilon);

}



