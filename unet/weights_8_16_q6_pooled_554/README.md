# [8,16] 六位量化 U-Net 权重

由已完成 QAT 的最佳模型导出，详见 weights_manifest.json；不是浮点影子参数。
将 unet_weights_data.cpp 加入 HLS 工程，包含 unet_weights.h。
数组在 namespace unet_q6_weights 中，包含 7 层卷积的 kernel/bias，共 4696 参数。
默认 weight_t=ap_fixed<6,1,AP_RND_CONV,AP_SAT>，值域 [-1,0.96875]，步长 1/32。
累加器 accum_t=ap_fixed<24,12>。裸 ap_fixed<6,1> 的默认舍入/溢出语义不同。
如需无 HLS 头文件的主机测试，所有翻译单元定义 UNET_Q6_FLOAT_REFERENCE。

卷积 kernel 由 Keras HWIO 转为 OIHW，未翻转空间核；final_kernel 使用
[20][8][1][1][1]，额外单例维度用于匹配现有 FinalConv2D 接口。
原始位编码：signed_code=round(value*32)，6 位二补码为 signed_code & 63。
数值字面量在 Q1.5 网格上精确表示，无二次有损量化。

输入原始 pooled energy 的 WHD=(5,5,4)，先 log1p(energy)/8，再六位量化。
PyTorch 风格输入以深度作为通道；若输入按 WHD 存放，应转置为 DHW=(4,5,5)。
模型是 2D Conv，特征通道 [8,16]，无 GroupNorm；其余处理必须包括：
- 普通卷积也必须加对应 bias，然后 ReLU 和六位饱和量化；
- MaxPool 5x5->2x2；nearest 2x2->5x5 的索引是 [0,0,0,1,1]；
- skip 拼接先 8 通道、后 16 通道；每层输出均六位量化；
- final 卷积输出 20 通道，经六位饱和后按 class*4+depth 恢复；
- 分类直接对 5 类 logits 取 argmax；概率可在软件执行 Softmax。
权重文件不包含 log1p 的硬件实现，也不自动修改旧 unet 推理代码。
