### omega-ai-v4-gpu @2024-05-20
- 新增：新增循环神经网络LSTM模型实现（小说生成器demo）
- 新增：新增循环神经网络seq2seq模型实现（中英文翻译器demo）
- 新增：新增transformer家族GPT模型支持，新增MultHeadSelfAttention（多头自注意力机制）实现FastCausalSelfAttentionLayer、MultiHeadAttentionLayer，新增MLP层实现MLPLayer，新增EmbeddingIDLayer（输入数据为id），新增Layer Normallization层等transformer系列基础层
- 新增：新增大语言nano GPT2模型实现（莎士比亚剧本生成demo）
- 新增：新增大语言GPT2模型实现（中文聊天机器人demo）
- 新增：新增大语言GPT2模型实现（中文医疗问答系统demo）
- 新增：新增BPE（byte pair encode）tokenizer编码器实现

### omega-ai-v4-gpu @2023-12-01
- 新增：新增yolov4版本实现，具体结构请查看yolov4-tiny.cfg文件
- 新增：新增yolov7版本实现，添加yolov7 loss实现,具体理论解析请查看readme.md文件
- 新增：新增基于yolov7-tiny实现智能冰柜商品识别demo
- 新增：SiLU激活函数实现
- 新增：修改yoloLayer(yolo层)，根据yolov4版本实现scale缩放公式从原来exp(xy)+b修改成sigmoid(xy) * scale - 0.5 * (scale - 1)，该操作可一定程度减缓由于exp()函数带来的数值不稳定和无穷大NaN的现象
- 新增：新增GAN实现，详情源码请查看com.omega.gan包，里面实现了手写体数字生成与动漫头像生成的事例
- 新增：新增RNN循环神经网络模型实现，添加RNNBlockLayer层，该层实现了RNN,LSTM,GRU三种循环神经网络基础模块
- 后续：后续版本将逐渐实现引擎对CycleGAN风格迁移,LSTM,GRU,transformer等模型支持

### omega-ai-v4-gpu @2023-08-02
- 新增：新增自动求导功能(包含cpu，gpu版本)
- 新增：新增multiLabel_soft_margin loss损失函数，yolo loss（Yolov3Loss）
- 新增：新增yolov3目标识别实现，当前实现的yolo版本为yolov3版本(实现源码请移步YoloV3Test.java)
- 新增：新增目标识别数据增强功能(随机裁剪边缘，随机上下反转，hsv变换等)
- 新增：使用自动求导功能实现MSN损失函数，代替原有的MSN loss
- 后续：后续版本将逐渐实现引擎对yolov5,GAN,transformer等模型支持

### omega-ai-v4-gpu @2023-04-13
- 新增：omega-ai-v4-gpu版本添加cudnn支持，整体推理与训练效率提升4倍
- 优化：优化bn层，激活函数层内存使用，整体内存显存占用减少30%~40%
- 新增：新增yolo目标识别实现，当前实现的yolo版本为yolov1版本(实现源码请移步YoloV1Test.java)
- 新增：新增图片绘制工具，帮助绘制预测框与回显图片
- 后续：后续版本将逐渐实现引擎对yolov3,yolov5等模型支持

### omega-ai-v4-gpu @2023-01-10
- 新增：开启omega-ai-v4-gpu版本开发，该版本将实现对omega-ai的CUDNN全面支持
- 新增：新增全局平均池化层实现
- 新增：将softmax与cross_entropy结合成softmax_with_cross_entropy作为损失函数使用(注意:使用softmax_with_cross_entropy损失函数,将不需要额外添加SoftmaxLayer)
- 新增：新增BN层对CUDNN支持，实现源码请移步(实现源码请移步BNCudnnKernel.java)
- 后续：后续版本将逐渐实现引擎对CUDNN支持

### omega-ai-v3-gpu @2022-09-02
- 优化：修改bn层计算dmean公式,减少计算量
- 优化：更换数据存储方式，以便使用gpu计算，减少4维数组与1维数组之间的转换，获得成倍的计算效率提升
- 优化：全面优化gpu计算，更新cuda核函数实现，使得训练与预测计算效获得大大提升
- 后续：后续版本将进一步优化gpu版本，预计将整个计算过程搬迁入gpu计算，从而减少主机与设备(显卡)之间传输，希望进一步获得更快的计算速度

### omega-ai-v3-gpu @2022-08-17
- 新增：初步完成卷积层的gpu改造，使得卷积神经网络计算速度整体提升，增加im2col与col2im两个经典的核函数（Im2colKernel.cu，Col2imKernel.cu）
- 新增：添加cuda内存管理器，用于管理整体显存的生命周期，减少频繁申请显存的操作，减少主机与显卡之间的数据传输

### omega-ai-v3-gpu @2022-07-02
- 新增：开启omega-ai-v3-gpu版本开发，该版本将实现对omega-ai的gpu全面支持
- 优化：全面优化卷积层计算，包括前向传播与反向传播

### omega-ai-v3 @2022-06-20
- 新增：添加gup支持，使用jcuda调用cuda的cublasSgemm矩阵乘法，参考了caffe的卷积操作已将卷积操作优化成im2col+gemm实现，计算效率得到大大提高
- 新增：添加vgg16 demo，该模型在cifar10数据集上表现为测试数据集准确率86.45%
- 优化：利用jdk ForkJoin框架实现任务拆分，充分利用cpu多线程，提高对数组操作与计算速度
- 新增：参考darknet对学习率更新机制进行升级，目前已支持RANDOM、POLY、STEP、EXP、SIG等多种学习率更新方法，并且实现学习率warmup功能
- 新增：添加basicblock模块，新增resnet模型支持，目前该模型在cifar10数据集上的表现，epoch:300，测试数据集准确率为91.23%