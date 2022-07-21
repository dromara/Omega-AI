
# 自己打造一个深度学习框架 for java

###  前言
从16年开始利用空余时间研究深度学习的方面，由于工作的原因，最熟悉的编程语言就是java，所以框架的编程语言自然而然就使用了java。自己打造框架的初衷就是为了更加深入了解各个算法、模型、实现的原理和思路。
## 框架介绍
Omega-AI：基于java打造的深度学习框架，帮助你快速搭建神经网络，实现训练或测试模型，支持多线程运算，框架目前支持BP神经网络和卷积神经网络的构建。
### 源码地址：

[https://gitee.com/iangellove/omega-ai](https://gitee.com/iangellove/omega-ai)

[https://github.com/iangellove/Omega-AI](https://github.com/iangellove/Omega-AI)

### 版本更新
#### omega-engine-1.0.3

1.添加gup支持，使用jcuda调用cuda的cublasSgemm矩阵乘法，参考了caffe的卷积操作已将卷积操作优化成im2col+gemm实现，计算效率得到大大提高

2.添加vgg16 demo，该模型在cifar10数据集上表现为测试数据集准确率78.2%

3.利用jdk ForkJoin框架实现任务拆分，充分利用cpu多线程，提高对数组操作与计算速度

4.参考darknet对学习率更新机制进行升级，目前已支持RANDOM、POLY、STEP、EXP、SIG等多种学习率更新方法，并且实现学习率warmup功能

### 依赖
由于omega-engine-1.0.3加入了jcuda支持，所以1.0.3需要安装与jcuda版本对应的cuda，我在该项目中使用的是jcuda-11.2.0版本的包，那么我cuda需要安装11.2.x版本

### Demo展示
[基于卷积神经网络mnist手写数字识别](http://120.237.148.121:8011/mnist)
![在这里插入图片描述](https://img-blog.csdnimg.cn/b9b5846af6624bdf8f5d570c5052bc64.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTMyODMzMDQ=,size_1,color_FFFFFF,t_70#pic_center)
##  功能介绍
#### 支持的网络层类型：

Fullylayer 全连接层

ConvolutionLayer 卷积层

PoolingLayer 池化层

#### 激活函数层

SoftmaxLayer (softmax激活函)

ReluLayer

LeakyReluLayer

TanhLayer

SigmodLayer

#### 归一化层

BNLayer (Batch Normalization)

DropoutLayer

#### 优化器

Momentum

Adam

#### 训练器

BGDOptimizer (批量梯度下降法)

MBSGDOptimizer (小批量随机梯度下降)

SGDOptimizer（随机梯度下降算法）

#### 损失函数(loss function)

SquareLoss (平方差损失函数)

CrossEntropyLoss (交叉熵损失函数)

#### 学习率更新器（LearnRateUpdate）

NONE (固定学习率)

LR_DECAY (decay)

GD_GECAY (gd_decay)

CONSTANT(gd_decay)

RANDOM [Math.pow(RandomUtils.getInstance().nextFloat(), power) * this.lr]

POLY [this.lr * Math.pow((1.0f - (batchIndex * 1.0f / trainTime / dataSize * batchSize)), power)]

STEP [this.lr * Math.pow(this.scale, batchIndex / step)]

EXP [this.lr * Math.pow(this.gama, batchIndex)]

SIG [this.lr / (1 + Math.pow(Math.E, this.gama * (batchIndex - step)))]

#### 数据加载器

.bin (二进制数据文件)

.idx3-ubyte

.txt

## 使用说明

### 自带的数据集

iris（鸢尾花数据集）

mnist（手写数字数据集）

cifat_10 （cifat_10数据集）

### 数据集成绩

iris 训练次数8   测试数据集准确率100%

mnist 训练次数8 测试数据集准确率98.6% 

cifat_10 训练次数8 测试数据集准确率76.6%

## 事例代码

#### bp iris demo

```java
public void bpNetwork_iris() {
		// TODO Auto-generated method stub

		/**
		 * 读取训练数据集
		 */
		String iris_train = "/dataset/iris/iris.txt";
		
		String iris_test = "/dataset/iris/iris_test.txt";
		
		String[] labelSet = new String[] {"1","-1"};
		
		DataSet trainData = DataLoader.loalDataByTxt(iris_train, ",", 1, 1, 4, 2,labelSet);
		DataSet testData = DataLoader.loalDataByTxt(iris_test, ",", 1, 1, 4, 2,labelSet);
		
		System.out.println("train_data:"+JsonUtils.toJson(trainData));
	
		BPNetwork netWork = new BPNetwork(new SoftmaxWithCrossEntropyLoss());
		
		InputLayer inputLayer = new InputLayer(1,1,4);
		
		FullyLayer hidden1 = new FullyLayer(4, 40);
		
		ReluLayer active1 = new ReluLayer();
		
		FullyLayer hidden2 = new FullyLayer(40, 20);
		
		ReluLayer active2 = new ReluLayer();
		
		FullyLayer hidden3 = new FullyLayer(20, 2);

		SoftmaxWithCrossEntropyLayer hidden4 = new SoftmaxWithCrossEntropyLayer(2);
		
		netWork.addLayer(inputLayer);
		netWork.addLayer(hidden1);
		netWork.addLayer(active1);
		netWork.addLayer(hidden2);
		netWork.addLayer(active2);
		netWork.addLayer(hidden3);
		netWork.addLayer(hidden4);

		try {
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 8, 0.00001d, 10, LearnRateUpdate.NONE);
		
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
````

#### cnn mnist demo

```java
public void cnnNetwork_mnist() {
		// TODO Auto-generated method stub
		
		try {

			/**
			 * 读取训练数据集
			 */
			String mnist_train_data = "/dataset/mnist/train-images.idx3-ubyte";
			
			String mnist_train_label = "/dataset/mnist/train-labels.idx1-ubyte";
			
			String mnist_test_data = "/dataset/mnist/t10k-images.idx3-ubyte";
			
			String mnist_test_label = "/dataset/mnist/t10k-labels.idx1-ubyte";
			
			String[] labelSet = new String[] {"0","1","2","3","4","5","6","7","8","9"};
			
			Resource trainDataRes = new ClassPathResource(mnist_train_data);

			Resource trainLabelRes = new ClassPathResource(mnist_train_label);
			
			Resource testDataRes = new ClassPathResource(mnist_test_data);
			
			Resource testLabelRes = new ClassPathResource(mnist_test_label);
			
			DataSet trainData = DataLoader.loadDataByUByte(trainDataRes.getFile(), trainLabelRes.getFile(), labelSet, 1, 1 , 784, true);
			
			DataSet testData = DataLoader.loadDataByUByte(testDataRes.getFile(), testLabelRes.getFile(), labelSet, 1, 1 , 784, true);

			int channel = 1;
			
			int height = 28;
			
			int width = 28;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.momentum);
			
			netWork.learnRate = 0.001d;
			
			InputLayer inputLayer = new InputLayer(channel, 1, 784);
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 6, width, height, 5, 5, 2, 1, false);
			
			BNLayer bn1 = new BNLayer();
			
			LeakyReluLayer active1 = new LeakyReluLayer();
			
			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			ConvolutionLayer conv2 = new ConvolutionLayer(pool1.oChannel, 12, pool1.oWidth, pool1.oHeight, 5, 5, 0, 1, false);
			
			BNLayer bn2 = new BNLayer();
			
			LeakyReluLayer active2 = new LeakyReluLayer();
			
			DropoutLayer drop1 = new DropoutLayer(0.5d);
			
			
			PoolingLayer pool2 = new PoolingLayer(conv2.oChannel, conv2.oWidth, conv2.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);

			int fInputCount = pool2.oChannel * pool2.oWidth * pool2.oHeight;
			
			int inputCount = (int) (Math.sqrt((fInputCount) + 10) + 10);
			
			FullyLayer full1 = new FullyLayer(fInputCount, inputCount, false);

			BNLayer bn3 = new BNLayer();
			
			LeakyReluLayer active3 = new LeakyReluLayer();
			
			FullyLayer full2 = new FullyLayer(inputCount, 10);
			
			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);

			netWork.addLayer(inputLayer);
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			netWork.addLayer(pool1);
			netWork.addLayer(conv2);
			netWork.addLayer(bn2);
			netWork.addLayer(active2);
			netWork.addLayer(drop1);
			netWork.addLayer(pool2);
			netWork.addLayer(full1);
			netWork.addLayer(bn3);
			netWork.addLayer(active3);
			netWork.addLayer(full2);
			netWork.addLayer(softmax);

			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 10, 0.0001d, 96, LearnRateUpdate.NONE);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");

			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
````
#### cnn cifar10 demo

```java
public void cnnNetwork_cifar10() {
		// TODO Auto-generated method stub
		
		try {

	    	String[] labelSet = new String[] {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
	    	
			String[] train_data_filenames = new String[] {
					"/dataset/cifar-10/data_batch_1.bin",
					"/dataset/cifar-10/data_batch_2.bin",
					"/dataset/cifar-10/data_batch_3.bin",
					"/dataset/cifar-10/data_batch_4.bin",
					"/dataset/cifar-10/data_batch_5.bin"
			};
			
			String test_data_filename = "/dataset/cifar-10/test_batch.bin";
			
			DataSet trainData = DataLoader.getImagesToDataSetByBin(train_data_filenames, 10000, 3, 32, 32, 10, true, labelSet);
	    	
			DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, true, labelSet);
			
			System.out.println("data is ready.");
			
			int channel = 3;
			
			int height = 32;
			
			int width = 32;
			
			CNN netWork = new CNN(new SoftmaxWithCrossEntropyLoss(), UpdaterType.adam);
			
			netWork.learnRate = 0.001d;
			
			InputLayer inputLayer = new InputLayer(channel, height, width);
			netWork.addLayer(inputLayer);
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 16, width, height, 3, 3, 1, 1,false);
			netWork.addLayer(conv1);
			
			BNLayer bn1 = new BNLayer();
			netWork.addLayer(bn1);
			
			ReluLayer active1 = new ReluLayer();
			netWork.addLayer(active1);

			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			netWork.addLayer(pool1);
			
			
			ConvolutionLayer conv3 = new ConvolutionLayer(pool1.oChannel, 32, pool1.oWidth, pool1.oHeight, 3, 3, 1, 1,false);
			netWork.addLayer(conv3);
			
			BNLayer bn3 = new BNLayer();
			netWork.addLayer(bn3);
			
			ReluLayer active3 = new ReluLayer();
			netWork.addLayer(active3);
			
			PoolingLayer pool2 = new PoolingLayer(conv3.oChannel, conv3.oWidth, conv3.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			netWork.addLayer(pool2);

			
			ConvolutionLayer conv4 = new ConvolutionLayer(pool2.oChannel, 64, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1,false);
			netWork.addLayer(conv4);
			
			BNLayer bn4 = new BNLayer();
			netWork.addLayer(bn4);
			
			ReluLayer active4 = new ReluLayer();
			netWork.addLayer(active4);
			
			PoolingLayer pool3 = new PoolingLayer(conv4.oChannel, conv4.oWidth, conv4.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			netWork.addLayer(pool3);


			int fInputCount = pool3.oChannel * pool3.oWidth * pool3.oHeight;
			
			FullyLayer full1 = new FullyLayer(fInputCount, 256, true);
			netWork.addLayer(full1);
			
			ReluLayer active9 = new ReluLayer();
			netWork.addLayer(active9);

			DropoutLayer drop1 = new DropoutLayer(0.5d);
			netWork.addLayer(drop1);

			FullyLayer full2 = new FullyLayer(full1.oWidth, 10, true);
			netWork.addLayer(full2);
			
			SoftmaxWithCrossEntropyLayer softmax = new SoftmaxWithCrossEntropyLayer(10);
			netWork.addLayer(softmax);
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 30, 0.001d, 64, LearnRateUpdate.NONE);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainData);
			
			optimizer.test(testData);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
````
## 未来可期

实现rcnn、rnn、yolo等算法

### 训练情况可视化

支持动态调参，可视化训练
![在这里插入图片描述](https://img-blog.csdnimg.cn/8bd006e4fc1442cfbc2d5e3682a2c5f1.png#pic_center)

### 彩蛋

## 基于神经网络+遗传算法实现AI赛车游戏

http://119.3.123.193:8011/AICar

## 欢迎打扰

### QQ：465973119
### 技术交流QQ群：119593195
### 电子邮箱：465973119@qq.com


