<!-- 这是目录树文件 -->

- **开始**
	- [框架介绍](/)
	- [CUDA安装](/doc/start/example)
    - [多卡训练配置](/doc/start/example)
	- [在SpringBoot环境运行](/doc/start/example)

- **API**
	- **神经网络**
      - [BPNetwork bp神经网络](/doc/api/bp-network.md)
      - [CNN 卷积神经网络](/doc/api/cnn.md)
      - [RNN 循环神经网络](/doc/api/rnn.md)
      - [LSTM 长短期记忆网络](/doc/api/lstm.md)
      - [Seq2Seq 序列到序列模型](/doc/api/seq2seq.md)
      - [GAN 对抗神经网络](/doc/api/gan.md)
      - [Resnet 残差网络](/doc/api/resnet.md)
      - [VGG vgg神经网络](/doc/api/vgg.md)
      - [Unet U形网络](/doc/api/unet.md)
      - [Yolo yolo实时目标检测模型](/doc/api/yolo.md)
      - [Transformer transformer模型](/doc/api/transformer.md)
      - [GPT 生成式预训练模型](/doc/api/gpt.md)
      - [LLama 生成式预训练模型](/doc/api/llama.md)
      - [LLava llama多模态模型](/doc/api/llava.md)
      - [Diffusion 扩散模型](/doc/api/diffusion.md)
      - [VAE 变分自编码器](/doc/api/vae.md)
      - [Stable Diffusion 稳定扩散模型](/doc/api/stable-diffusion.md)

    - **网络层**
      - [Fullylayer 全连接层](/doc/api/fullylayer.md)
      - [ConvolutionLayer 卷积层](/doc/api/convolutionlayer.md)
      - [ConvolutionTransposeLayer 反卷积层](/doc/api/convolutiontransposelayer.md)
      - [PoolingLayer 池化层(maxpooling,meanpooling)](/doc/api/poolinglayer.md)
      - [AVGPooingLayer 全局平均池化层](/doc/api/avgpoolinglayer.md)
      - [AdaptiveAvgPool2DLayer 全局平均池化层2D](/doc/api/adaptiveavgpool2dlayer.md)
      - [EmbeddingIDLayer 向量映射层(将高维度词向量映射成低维度向量)](/doc/api/embeddingidlayer.md)
      - [RNNLayer 循环神经网络层](/doc/api/rnnlayer.md)
      - [LSTMLayer 长短记忆网络层](/doc/api/lstmlayer.md)
      - [RouteLayer 路由层](/doc/api/routelayer.md)
      - [UPSampleLayer 上采样层](/doc/api/upsamplelayer.md)
      - [YoloLayer yolo层](/doc/api/yololayer.md)
      - [FastCausalSelfAttentionLayer 多头自注意力层](/doc/api/fastcausalselfattentionlayer.md)
      - [LlamaCausalSelfAttentionLayer llama多头注意力层](/doc/api/llamacausalselfattentionlayer.md)
      - [MLPLayer gpt2 - mlp层](/doc/api/mlplayer.md)
      - [TransformerBlock transformer基础块](/doc/api/transformerblock.md)
      - [UNetCrossAttentionLayer 带交差注意力机制层](/doc/api/unetcrossattentionlayer.md)
      - [UNetDownBlock Unet下采样网络块](/doc/api/unetdownblock.md)
      - [UNetMidBlock Unet中间网络块](/doc/api/unetmidblock.md)
      - [UNetUpBlock Unet上采样网络块](/doc/api/unetupblock.md)
      - [TimeEmbeddingLayer 时间维度嵌入层](/doc/api/timeembeddinglayer.md)

  - **激活函数**
    - [ReluLayer](/doc/api/relulayer.md)
    - [LeakyReluLayer](/doc/api/leakyrelulayer.md)
    - [TanhLayer](/doc/api/tanhlayer.md)
    - [SigmodLayer](/doc/api/sigmodlayer.md)
    - [SiLULayer](/doc/api/silulayer.md)
    - [GeLULayer](/doc/api/gelulayer.md)

  - **归一化**
    - [BNLayer (Batch Normalization) 批归一化](/doc/api/bnlayer.md)
    - [LNLayer (Layer Normalization) 层归一化](/doc/api/lnlayer.md)
    - [InstanceNormaliztionLayer 实例归一化](/doc/api/instancenormalizationlayer.md)
    - [RMSLayer 均方根归一化](/doc/api/rmslayer.md)

  - **正则化**
    - [DropoutLayer](/doc/api/dropoutlayer.md)

  - **优化器**
    - [Momentum](/doc/api/momentum.md)
    - [Sgd](/doc/api/sgd.md)
    - [Adam](/doc/api/adam.md)
    - [Adamw](/doc/api/adamw.md)
    - [RMSProp](/doc/api/rmsprop.md)

  - **训练器**
    - [MBSGDOptimizer (小批量随机梯度下降)](/doc/api/mbsgdoptimizer.md)

  - **损失函数**
    - [MSELoss (平方差损失函数)](/doc/api/mseloss.md)
    - [CrossEntropyLoss (交叉熵损失函数)](/doc/api/crossentropyloss.md)
    - [CrossEntropyLossWithSoftmax (交叉熵损失 + softmax)](/doc/api/crossentropylosswithsoftmax.md)
    - [MultiLabelSoftMargin (多标签损失函数)](/doc/api/multilabelsoftmargin.md)

  - **学习率更新器**
    - [CONSTANT (固定学习率)](/doc/api/constant.md)
    - [LR_DECAY (decay)](/doc/api/lr-decay.md)
    - [GD_GECAY (gd_decay)](/doc/api/gd-gecay.md)
    - [POLY](/doc/api/poly.md)
    - [EXP](/doc/api/exp.md)
    - [SIG](/doc/api/sig.md)

- **其它**
	- [更新日志](/doc/more/update-log) 
	- [框架生态](/doc/more/link) 
	- [框架博客](/doc/more/blog) 
	- [需求提交](/doc/more/demand-commit) 
	- [问卷调查](/doc/more/wenjuan)
	- [加入讨论群](/doc/more/join-group)
	- [赞助 Omega-AI](/doc/more/donate)

- **附录**
	- [常见问题排查](/doc/more/common-questions) 
    - [issue 提问模板](/doc/fun/issue-template)
	- [为mega-AI贡献代码](/doc/fun/git-pr)
	- [Omega-AI框架掌握度--在线考试](/doc/fun/omega-ai-test)
	- [团队成员](/doc/fun/team)
	


<br/><br/><br/><br/><br/><br/><br/>
<p style="text-align: center;">----- 到底线了 -----</p>