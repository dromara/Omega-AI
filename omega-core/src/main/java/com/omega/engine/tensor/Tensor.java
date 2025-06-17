package com.omega.engine.tensor;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.Graph;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.ad.op.gpu.OPKernel;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.parallel.ddp.distributed.SerializablePointer;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.cudaStream_t;

import java.io.Serializable;
import java.util.UUID;

/**
 * 四维张量类，用于表示如 [batch, channel, height, width] 格式的数据结构。
 * 通常用于卷积神经网络的输入或中间层数据。
 */
public class Tensor implements Serializable {
    private static final long serialVersionUID = 5844762745177624845L;
    public String id;
    //个数
//    private int number= 0;
    //通道
//    private int channel = 0;
    //高度
//    private int height = 0;
    //宽度
//    private int width = 0;
    //数据长度= number * channel * height * width
    private int dataLength = 0;
    //GPU 中实际分配的数据长度
    private int gpuLength = 0;
    //数据
    private float[] data;
    // 临时缓存区，用于按样本提取
    private float[] once;
    //GPU 数据指针，指向 CUDA 内存地址
    private Pointer gpuData;
    //可共享的 GPU 数据指针（用于多线程/分布式通信）
    private SerializablePointer shareGPU;
    //是否启用 GPU 加速
    private boolean hasGPU = false;
    //是否需要计算梯度（用于反向传播）
    private boolean requiresGrad = false;
    //当前张量的梯度张量
    private Tensor grad;
    //所属计算图，用于自动微分与执行优化
    private Graph g;
    //当前形状记录，用于 view 等方法
    private int[] shape;
    //原始形状记录，用于 viewOrg 等恢复原始形状的方法
    private int[] orgShape;
    //临时张量，用于中间计算缓存
    private Tensor tmp;
    //单元素临时张量，常用于标量结果存储
    private Tensor tmp_once;

    /**
     * 计算给定形状的元素总数。
     *
     * @param shape 形状数组
     * @return 元素总数
     */
    private int getDataLengthByShape(int[] shape) {
        int total = 1;
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("维度必须大于0: " + dim);
            }
            total *= dim;
        }

        return total;
    }

    /**
     * 用于创建一个张量对象。
     *
     * @param data         初始数据数组，如果为 null 且 onlyGPU 为 false，则会分配新的内存。
     * @param defaultValue 如果 data 为 null 且 onlyGPU 为 false，则使用该默认值初始化数据。
     * @param hasGPU       是否启用 GPU 进行计算。
     * @param onlyGPU      是否仅在 GPU 上存储数据（不保留 CPU 数据副本）。
     * @param g            所属的计算图，用于自动微分。
     * @param shape        张量的形状，以可变参数形式传入（例如 number, channel, height, width）。
     */
    public Tensor(float[] data, Float defaultValue, boolean hasGPU, boolean onlyGPU, Graph g, int... shape) {
        int dataLength = getDataLengthByShape(shape);
        this.gpuLength = dataLength;
        this.g = g;
        if (defaultValue != null) {
            this.data = MatrixUtils.val(dataLength, defaultValue);
        }
        this.data = data == null ?  new float[dataLength] : data.clone();
        this.orgShape = shape;
        this.shape = shape;
        this.setHasGPU(hasGPU);
        if (hasGPU) {
            gpuData = CUDAMemoryManager.getPointer(dataLength);
            JCuda.cudaMemcpy(gpuData, Pointer.to(this.data), dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaDeviceSynchronize();
            if (onlyGPU) {
                this.data = null;
            }
        } else {
            this.clearGPU();
        }
    }


    /**
     * 创建一个张量对象。
     *
     * @param shape 张量的形状，以可变参数形式传入（例如 number, channel, height, width）。
     */
    public Tensor(int... shape) {
        this(null, null, false, false, null, shape);
    }

    /**
     * 创建一个带有默认值的张量对象。
     *
     * @param defaultValue 默认值，用于初始化数据。
     * @param shape        张量的形状，以可变参数形式传入（例如 number, channel, height, width）。
     */
    public Tensor(Float defaultValue, int... shape) {
        this(null, defaultValue, false, false, null, shape);
    }

    /**
     * 创建一个带有默认值和 GPU 支持的张量对象。
     *
     * @param defaultValue 默认值，用于初始化数据。
     * @param hasGPU       是否启用 GPU 进行计算。
     * @param shape        张量的形状，以可变参数形式传入（例如 number, channel, height, width）。
     */
    public Tensor(Float defaultValue, boolean hasGPU, int... shape) {
        this(null, defaultValue, hasGPU, false, null, shape);
    }

    /**
     * 创建一个带有初始数据和 GPU 支持的张量对象。
     *
     * @param data   初始数据数组。
     * @param hasGPU 是否启用 GPU 进行计算。
     * @param shape  张量的形状，以可变参数形式传入（例如 number, channel, height, width）。
     */
    public Tensor(float[] data, boolean hasGPU, int... shape) {
        this(data, null, hasGPU, false, null, shape);
    }

    /**
     * 创建一个仅在 GPU 上存储数据的张量。
     * 若 onlyGPU 为 true，则不分配 CPU 内存。
     *
     * @param number  样本数
     * @param channel 通道数
     * @param height  高度
     * @param width   宽度
     */
    @Deprecated
    public Tensor(int number, int channel, int height, int width) {
        this(null, null, false, false, null, number, channel, height, width);
    }


    /**
     * 创建一个具有指定形状和 GPU 支持的张量。
     *
     * @param number  样本数
     * @param channel 通道数
     * @param height  高度
     * @param width   宽度
     * @param hasGPU  是否启用 GPU 加速
     */
    @Deprecated
    public Tensor(int number, int channel, int height, int width, boolean hasGPU) {
        this(null, null, hasGPU, false, null, number, channel, height, width);
    }

    /**
     * 创建一个仅在 GPU 上存储数据的张量（不保留 CPU 数据副本）。
     *
     * @param number  样本数
     * @param channel 通道数
     * @param height  高度
     * @param width   宽度
     * @param hasGPU  是否启用 GPU 加速
     * @param onlyGPU 是否仅在 GPU 上分配内存，不保留 CPU 数据
     */
    @Deprecated
    public Tensor(int number, int channel, int height, int width, boolean hasGPU, boolean onlyGPU) {
        this(null, null, hasGPU, onlyGPU, null, number, channel, height, width);
    }

    /**
     * 创建一个与计算图关联的张量。
     *
     * @param number  样本数
     * @param channel 通道数
     * @param height  高度
     * @param width   宽度
     * @param hasGPU  是否启用 GPU 加速
     * @param g       所属的计算图，用于自动微分
     */
    @Deprecated
    public Tensor(int number, int channel, int height, int width, boolean hasGPU, Graph g) {
        this(null, null, hasGPU, false, g, number, channel, height, width);
    }

    /**
     * 使用提供的数据初始化张量。
     *
     * @param number  样本数
     * @param channel 通道数
     * @param height  高度
     * @param width   宽度
     * @param data    初始数据数组
     */
    @Deprecated
    public Tensor(int number, int channel, int height, int width, float[] data) {
        this(data, null, false, false, null, number, channel, height, width);
    }

    /**
     * 使用提供的数据初始化张量并设置 GPU 支持。
     *
     * @param number  样本数
     * @param channel 通道数
     * @param height  高度
     * @param width   宽度
     * @param data    初始数据数组
     * @param hasGPU  是否启用 GPU 加速
     */
    @Deprecated
    public Tensor(int number, int channel, int height, int width, float[] data, boolean hasGPU) {
        this(data, null, hasGPU, false, null, number, channel, height, width);
    }

    /**
     * 使用提供的数据初始化张量，并尝试共享 GPU 内存。
     *
     * @param number  样本数
     * @param channel 通道数
     * @param height  高度
     * @param width   宽度
     * @param data    初始数据数组
     * @param hasGPU  是否启用 GPU 加速
     * @param share
     */
    @Deprecated
    public Tensor(int number, int channel, int height, int width, float[] data, boolean hasGPU, boolean share) {
        this(data, null, hasGPU, false, null, number, channel, height, width);
    }

    /**
     * 使用提供的数据初始化张量并与计算图关联。
     *
     * @param number  样本数
     * @param channel 通道数
     * @param height  高度
     * @param width   宽度
     * @param data    初始数据数组
     * @param hasGPU  是否启用 GPU 加速
     * @param g       所属的计算图，用于自动微分
     */
    @Deprecated
    public Tensor(int number, int channel, int height, int width, float[] data, boolean hasGPU, Graph g) {
        this(data, null, hasGPU, false, g, number, channel, height, width);
    }

    /**
     * 使用整数值初始化张量（自动转换为浮点型）。
     *
     * @param number  样本数
     * @param channel 通道数
     * @param height  高度
     * @param width   宽度
     * @param val     用于初始化的整数值
     * @param hasGPU  是否启用 GPU 加速
     */
    @Deprecated
    public Tensor(int number, int channel, int height, int width, int val, boolean hasGPU) {
        this(null, Float.valueOf(val), hasGPU, false, null, number, channel, height, width);
    }

    /**
     * 使用整数值初始化张量并与计算图关联。
     *
     * @param number  样本数
     * @param channel 通道数
     * @param height  高度
     * @param width   宽度
     * @param val     用于初始化的整数值
     * @param hasGPU  是否启用 GPU 加速
     * @param g       所属的计算图，用于自动微分
     */
    @Deprecated
    public Tensor(int number, int channel, int height, int width, int val, boolean hasGPU, Graph g) {
        this(null, Float.valueOf(val), hasGPU, false, g, number, channel, height, width);
    }

    public static Tensor createTensor(Tensor t, int number, int channel, int height, int width, float[] data, boolean hasGPU) {
        if (t == null) {
            t = new Tensor(number, channel, height, width, data, hasGPU);
        } else {
            t.resize(number, channel, height, width, data);
            t.orgShape = new int[]{number, channel, height, width};
        }
        return t;
    }

    public static Tensor createTensor(Tensor t, int number, int channel, int height, int width, boolean hasGPU) {
        if (t == null) {
            t = new Tensor(number, channel, height, width, hasGPU);
        } else {
            t.resize(number, channel, height, width);
            t.orgShape = new int[]{number, channel, height, width};
        }
        return t;
    }

    public static Tensor createGPUTensor(Tensor t, int number, int channel, int height, int width, boolean hasGPU) {
        if (t == null) {
            t = new Tensor(number, channel, height, width, hasGPU, true);
        } else {
            t.resize(number, channel, height, width, true);
            t.orgShape = new int[]{number, channel, height, width};
        }
        //		System.err.println("in-create");
        return t;
    }

    public static Tensor createGPUTensor(Tensor t, int[] shape, boolean hasGPU) {
        if (t == null) {
            t = new Tensor(shape[0], shape[1], shape[2], shape[3], hasGPU, true);
        } else {
            t.resize(shape[0], shape[1], shape[2], shape[3], true);
            t.orgShape = new int[]{shape[0], shape[1], shape[2], shape[3]};
        }
        //		System.err.println("in-create");
        return t;
    }

    public Tensor getTmp() {
        if (tmp == null) {
            tmp = new Tensor(getShape()[0], getShape()[1], getShape()[2], getShape()[3], hasGPU);
        }
        return tmp;
    }

    public Tensor getTmpOnce() {
        if (tmp_once == null) {
            tmp_once = new Tensor(1, 1, 1, 1, true);
        }
        return tmp_once;
    }

    public String getId() {
        if (this.id == null) {
            this.id = UUID.randomUUID().toString();
        }
        return this.id;
    }

    public Tensor copy() {
        float[] dest = new float[getDataLength()];
        System.arraycopy(getData(), 0, dest, 0, getDataLength());
        Tensor dis = new Tensor(getShape()[0], getShape()[1], getShape()[2], getShape()[3], dest, hasGPU);
        return dis;
    }

    public Tensor copyGPU() {
        float[] dest = new float[getDataLength()];
        System.arraycopy(this.syncHost(), 0, dest, 0, getDataLength());
        Tensor dis = new Tensor(getShape()[0], getShape()[1], getShape()[2], getShape()[3], dest, hasGPU);
        return dis;
    }

    public void copy(Tensor tmp) {
        System.arraycopy(this.syncHost(), 0, tmp.getData(), 0, getDataLength());
        if (tmp.hasGPU) {
            tmp.hostToDevice();
        }
    }

    public void copyGPU(Tensor tmp) {
        JCuda.cudaMemcpy(tmp.getGpuData(), gpuData, this.getDataLength() * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToDevice);
    }

    public Tensor createLike() {
        return new Tensor(getShape()[0], getShape()[1], getShape()[2], getShape()[3], hasGPU);
    }

    public Tensor createGPULike() {
        // return new Tensor(1, 1, 1, 1, hasGPU, true);
        return new Tensor(null, null, hasGPU, true, null, shape);
    }

    public Tensor createLike(float value) {
        return new Tensor(getShape()[0], getShape()[1], getShape()[2], getShape()[3], MatrixUtils.val(this.getDataLength(), value), hasGPU);
    }

    public void resize(int number, int channel, int height, int width) {
        this.getShape()[0] = number;
        this.getShape()[1] = channel;
        this.getShape()[2] = height;
        this.getShape()[3] = width;
        this.dataLength = number * channel * height * width;
        this.data = new float[this.getDataLength()];
        if (hasGPU) {
            if (gpuData != null) {
                CUDAMemoryManager.free(gpuData);
            }
            gpuData = CUDAMemoryManager.getPointer(getDataLength());
        }
    }

    public void resize(int number, int channel, int height, int width, boolean onlyGPU) {
        this.getShape()[0] = number;
        this.getShape()[1] = channel;
        this.getShape()[2] = height;
        this.getShape()[3] = width;
        this.dataLength = number * channel * height * width;
        this.gpuLength = this.getDataLength();
        if (!onlyGPU) {
            this.data = new float[this.getDataLength()];
        }
        if (hasGPU) {
            if (gpuData != null) {
                CUDAMemoryManager.free(gpuData);
            }
            gpuData = CUDAMemoryManager.getPointer(getDataLength());
            JCuda.cudaDeviceSynchronize();
        }
    }

    public void resize(int number, int channel, int height, int width, float[] data) {
        this.getShape()[0] = number;
        this.getShape()[1] = channel;
        this.getShape()[2] = height;
        this.getShape()[3] = width;
        this.dataLength = number * channel * height * width;
        this.data = data;
        if (hasGPU) {
            if (gpuData != null) {
                CUDAMemoryManager.free(gpuData);
            }
            gpuData = CUDAMemoryManager.getPointer(getDataLength());
            JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.getDataLength() * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public void copy(int n, float[] dest) {
        if (n < getShape()[0]) {
            System.arraycopy(getData(), n * getShape()[1] * getShape()[2] * getShape()[3], dest, 0, getShape()[1] * getShape()[2] * getShape()[3]);
        } else {
            throw new RuntimeException("获取数据失败[下标超出长度].");
        }
    }

    public Tensor view(int number, int channel, int height, int width) {
        this.getShape()[0] = number;
        this.getShape()[1] = channel;
        this.getShape()[2] = height;
        this.getShape()[3] = width;
        return this;
    }

    public Tensor viewOrg(int number, int channel, int height, int width) {
        this.getShape()[0] = number;
        this.getShape()[1] = channel;
        this.getShape()[2] = height;
        this.getShape()[3] = width;
        this.dataLength = number * channel * height * width;
        this.orgShape[0] = number;
        this.orgShape[1] = channel;
        this.orgShape[2] = height;
        this.orgShape[3] = width;
        return this;
    }

    public Tensor view(int[] shape) {
        this.getShape()[0] = shape[0];
        this.getShape()[1] = shape[1];
        this.getShape()[2] = shape[2];
        this.getShape()[3] = shape[3];
        return this;
    }

    public Tensor viewOrg() {
        this.getShape()[0] = orgShape[0];
        this.getShape()[1] = orgShape[1];
        this.getShape()[2] = orgShape[2];
        this.getShape()[3] = orgShape[3];
        return this;
    }

    public int[] shape() {
        return new int[]{this.getShape()[0], this.getShape()[1], this.getShape()[2], this.getShape()[3]};
    }

    public void showShape() {
        System.out.println(JsonUtils.toJson(shape()));
    }

    public void showShape(String label) {
        System.out.println(label + ":" + JsonUtils.toJson(shape()));
    }

    public int getNumber() {
        return getShape()[0];
    }

    public void setNumber(int number) {
        this.getShape()[0] = number;
    }

    public int getChannel() {
        return getShape()[1];
    }

    public void setChannel(int channel) {
        this.getShape()[1] = channel;
    }

    public int getHeight() {
        return getShape()[2];
    }

    public void setHeight(int height) {
        this.getShape()[2] = height;
    }

    public int getWidth() {
        return getShape()[3];
    }

    public void setWidth(int width) {
        this.getShape()[3] = width;
    }

    public int getDataLength() {
        return getShape()[0] * getShape()[1] * getShape()[2] * getShape()[3];
    }

    public int getGpuLength() {
        return this.gpuLength;
    }

    public void setDataLength(int dataLength) {
        this.dataLength = dataLength;
    }

    public int getOnceSize() {
        return getShape()[1] * getShape()[2] * getShape()[3];
    }

    public int[] getShape() {
        return shape;
    }

    public float[] getData() {
        return this.data;
    }

    public void setData(float[] data) {
        this.data = data;
        if (isHasGPU()) {
            this.hostToDevice();
        }
    }

    public void setData(int[] data) {
        for (int i = 0; i < data.length; i++) {
            this.getData()[i] = data[i] * 1.0f;
        }
        if (isHasGPU()) {
            this.hostToDevice();
        }
    }

    public void copyData(float[] data) {
        System.arraycopy(data, 0, this.getData(), 0, data.length);
        if (isHasGPU()) {
            this.hostToDevice();
        }
    }

    public float getByIndex(int n, int c, int h, int w) {
        return this.getData()[n * getShape()[1] * getShape()[2] * getShape()[3] + c * getShape()[2] * getShape()[3] + h * getShape()[3] + w];
    }

    public float[] getByNumber(int n) {
        System.arraycopy(getData(), n * getShape()[1] * getShape()[2] * getShape()[3], getOnce(), 0, getShape()[1] * getShape()[2] * getShape()[3]);
        return this.once;
    }

    public float[] getByOffset(int start, int len) {
        float[] tmp = new float[len];
        System.arraycopy(getData(), start, tmp, 0, len);
        return tmp;
    }

    public void setByNumber(int n, float[] x) {
        System.arraycopy(x, 0, getData(), n * getShape()[1] * getShape()[2] * getShape()[3], getShape()[1] * getShape()[2] * getShape()[3]);
    }

    public void getByNumber(int n, float[] once) {
        if (once == null || once.length != getShape()[1] * getShape()[2] * getShape()[3]) {
            once = new float[getShape()[1] * getShape()[2] * getShape()[3]];
        }
        System.arraycopy(getData(), n * getShape()[1] * getShape()[2] * getShape()[3], once, 0, getShape()[1] * getShape()[2] * getShape()[3]);
    }

    public float[] getByNumberAndChannel(int n, int c) {
        int start = n * getShape()[1] * getShape()[2] * getShape()[3] + c * getShape()[2] * getShape()[3];
        System.arraycopy(getData(), start, getOnce(), 0, getShape()[2] * getShape()[3]);
        return this.once;
    }

    public void setByNumberAndChannel(int n, int c, float[] x) {
        System.arraycopy(x, 0, getData(), n * getShape()[1] * getShape()[2] * getShape()[3] + c * getShape()[2] * getShape()[3], getShape()[2] * getShape()[3]);
    }

    public void getByNumberAndChannel(int n, int c, float[] once) {
        if (once == null || once.length != getShape()[2] * getShape()[3]) {
            once = new float[getShape()[2] * getShape()[3]];
        }
        int start = n * getShape()[1] * getShape()[2] * getShape()[3] + c * getShape()[2] * getShape()[3];
        System.arraycopy(getData(), start, once, 0, getShape()[2] * getShape()[3]);
    }

    public void clear() {
        for (int i = 0; i < this.getDataLength(); i++) {
            this.getData()[i] = 0;
        }
    }

    public void val_cpu(float val) {
        for (int i = 0; i < this.getDataLength(); i++) {
            this.getData()[i] = val;
        }
    }

    public void clear(int number, int channel, int height, int width) {
        this.getShape()[0] = number;
        this.getShape()[1] = channel;
        this.getShape()[2] = height;
        this.getShape()[3] = width;
        this.dataLength = number * channel * height * width;
        this.data = new float[this.getDataLength()];
    }

    public Pointer getGpuData() {
        return gpuData;
    }

    public void setGpuData(Pointer gpuData) {
        this.gpuData = gpuData;
    }

    public float[] syncHost() {
        if (getData() == null || getData().length != this.getDataLength()) {
            this.data = new float[this.getDataLength()];
        }
        JCuda.cudaMemcpy(Pointer.to(getData()), gpuData, this.getDataLength() * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        return getData();
    }

    public void syncHost(float[] tmp) {
        JCuda.cudaMemcpy(Pointer.to(tmp), gpuData, this.getDataLength() * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
    }

    public void hostToDevice() {
        if (hasGPU) {
            if (gpuData == null) {
                gpuData = CUDAMemoryManager.getPointer(getDataLength());
            }
            JCuda.cudaMemcpy(gpuData, Pointer.to(getData()), this.getDataLength() * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public void hostToDevice(float[] data) {
        if (hasGPU) {
            if (gpuData == null) {
                gpuData = CUDAMemoryManager.getPointer(getDataLength());
            }
            JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.getDataLength() * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public void freeGPU() {
        if (gpuData != null) {
            JCuda.cudaFree(gpuData);
            gpuData = CUDAMemoryManager.getPointer(getDataLength());
        }
    }

    public void showDM() {
        syncHost();
        System.out.println(JsonUtils.toJson(getData()));
    }

    public void showDMAndShape() {
        syncHost();
        System.out.println(JsonUtils.toJson(shape()) + ":" + JsonUtils.toJson(getData()));
    }

    public void showDM(String label) {
        syncHost();
        System.out.println(label + "[" + getShape()[0] + ":" + getShape()[1] + ":" + getShape()[2] + ":" + getShape()[3] + "]" + JsonUtils.toJson(getData()));
    }

    public void showDMByNumber(int number) {
        syncHost();
        System.out.println(JsonUtils.toJson(this.getByNumber(number)));
    }

    public void checkDMByNumber(int number) {
        syncHost();
        System.out.println(MatrixUtils.isZero(this.getByNumber(number)));
    }

    public void checkDMZero() {
        syncHost();
        System.out.println(MatrixUtils.isZero(getData()));
    }

    public void showDMByOffset(int start, int len) {
        syncHost();
        System.out.println(JsonUtils.toJson(this.getByOffset(start, len)));
    }

    public void showDMByOffset(int start, int len, String label) {
        syncHost();
        System.out.println(label + JsonUtils.toJson(this.getByOffset(start, len)));
    }

    public void showDMByOffsetRed(int start, int len, String label) {
        syncHost();
        System.err.println(label + JsonUtils.toJson(this.getByOffset(start, len)));
    }

    public void showDM(int index) {
        syncHost();
        System.out.println(getData()[index]);
    }

    public void showDM(int index, String label) {
        syncHost();
        System.out.println(label + ":" + getData()[index]);
    }

    public boolean checkDM() {
        for (float val : syncHost()) {
            if (val > 0) {
                return true;
            }
        }
        return false;
    }

    public boolean checkNan() {
        return MatrixOperation.isNaN(syncHost());
    }

    public boolean checkInf() {
        return MatrixOperation.isInfinite(syncHost());
    }

    public void clearGPU(cudaStream_t stream) {
        checkCUDA(JCuda.cudaMemsetAsync(gpuData, 0, this.getDataLength() * (long) Sizeof.FLOAT, stream));
        //		checkCUDA(JCuda.cudaMemset(gpuData, 0, this.dataLength * Sizeof.FLOAT));
    }

    public void clearGPU() {
        if (gpuData != null) {
            checkCUDA(JCuda.cudaMemset(gpuData, 0, this.getDataLength() * (long) Sizeof.FLOAT));
            //			JCuda.cudaDeviceSynchronize();
        }
    }

    public void valueGPU(int val) {
        if (gpuData != null) {
            checkCUDA(JCuda.cudaMemset(gpuData, val, this.getDataLength() * (long) Sizeof.FLOAT));
            JCuda.cudaDeviceSynchronize();
        }
    }

    public void checkCUDA(int code) {
        if (code != cudaError.cudaSuccess) {
            throw new RuntimeException(cudaError.stringFor(code));
        }
    }

    public boolean isHasGPU() {
        return hasGPU;
    }

    public void setHasGPU(boolean hasGPU) {
        this.hasGPU = hasGPU;
    }

    public boolean isRequiresGrad() {
        return requiresGrad;
    }

    public void setRequiresGrad(boolean requiresGrad) {
        this.requiresGrad = requiresGrad;
        if (this.requiresGrad) {
            this.getGrad();
        }
    }

    public Tensor getGrad() {
        if (this.grad == null) {
            this.grad = new Tensor(getShape()[0], getShape()[1], getShape()[2], getShape()[3], this.hasGPU);
        }
        return grad;
    }

    public void setGrad(Tensor grad) {
        this.grad = grad;
    }

    public void setGrad(float[] grad) {
        if (this.grad == null) {
            this.grad = new Tensor(getShape()[0], getShape()[1], getShape()[2], getShape()[3], grad, this.hasGPU);
        } else {
            this.grad.data = grad;
            this.grad.hostToDevice();
        }
    }

    public Tensor getGrad(float[] val) {
        if (this.grad == null) {
            this.grad = new Tensor(getShape()[0], getShape()[1], getShape()[2], getShape()[3], val, this.hasGPU);
        }
        return grad;
    }

    public void setGrad(Tensor grad, int[] position, OPKernel kenel) {
        if (this.grad == null) {
            this.grad = new Tensor(getShape()[0], getShape()[1], getShape()[2], getShape()[3], this.hasGPU);
        }
        int dims = position[0];
        int start = position[1];
        int count = position[2];
        switch (dims) {
            case 0:
                if (isHasGPU()) {
                    kenel.copy_number_gpu(this.grad, grad, start, 1);
                } else {
                    setGradByNumber(grad.getData(), start, count);
                }
                break;
            case 1:
                if (isHasGPU()) {
                    kenel.copy_channel_gpu(this.grad, grad, start, 1);
                } else {
                    setGradByChannel(grad.getData(), start, count);
                }
                break;
            default:
                break;
        }
    }

    public void zeroGrad(OPKernel op) {
        if (this.grad != null) {
            this.grad.fill(0.0f, op);
        }
    }

    public void random() {
        RandomUtils.gaussianRandom(getData(), 1.0f);
        if (isHasGPU()) {
            this.hostToDevice();
        }
    }

    public boolean isZero() {
        if (isHasGPU()) {
            return MatrixUtils.isZero(this.syncHost());
        }
        return MatrixUtils.isZero(getData());
    }

    /**
     * tensor基础操作
     *
     * @return
     */
    public Tensor add(Tensor y) {
        return g.OP(OPType.add, this, y);
    }

    public Tensor add(float y) {
        return g.OP(OPType.add, this, y);
    }

    public Tensor sub(Tensor y) {
        return g.OP(OPType.subtraction, this, y);
    }

    public Tensor sub(float y) {
        return g.OP(OPType.subtraction, this, y);
    }

    public Tensor scalarSub(float scalar) {
        return g.OP(OPType.scalarSubtraction, this, scalar);
    }

    public Tensor mul(Tensor y) {
        return g.OP(OPType.multiplication, this, y);
    }

    public Tensor mul(float scalar) {
        return g.OP(OPType.multiplication, this, scalar);
    }

    public Tensor div(Tensor y) {
        return g.OP(OPType.division, this, y);
    }

    public Tensor div(float scalar) {
        return g.OP(OPType.division, this, scalar);
    }

    public Tensor scalarDiv(float scalar) {
        return g.OP(OPType.scalarDivision, this, scalar);
    }

    public Tensor maximum(Tensor y) {
        return g.OP(OPType.maximum, this, y);
    }

    public Tensor minimum(Tensor y) {
        return g.OP(OPType.minimum, this, y);
    }

    public Tensor dot(Tensor y) {
        return g.OP(OPType.dot, this, y);
    }

    public Tensor log() {
        return g.OP(OPType.log, this);
    }

    public Tensor transpose() {
        return g.OP(OPType.transpose, this);
    }

    public Tensor pow() {
        return g.OP(OPType.pow, this, 2.0f);
    }

    public float norm(TensorOP op) {
        getTmpOnce().valueGPU(0);
        op.pow(this, 2, getTmp());
        op.sum(getTmp(), getTmpOnce(), 0);
        op.sqrt(getTmpOnce(), getTmpOnce());
        //    	System.out.println("sqrt:"+getTmpOnce().syncHost()[0]);
        return getTmpOnce().syncHost()[0];
    }

    public Tensor pow(float scalar) {
        return g.OP(OPType.pow, this, scalar);
    }

    public Tensor sin() {
        return g.OP(OPType.sin, this);
    }

    public Tensor cos() {
        return g.OP(OPType.cos, this);
    }

    public Tensor tan() {
        return g.OP(OPType.tan, this);
    }

    public Tensor atan() {
        return g.OP(OPType.atan, this);
    }

    public Tensor exp() {
        return g.OP(OPType.exp, this);
    }

    public Tensor sum(int axis) {
        return g.OP(OPType.sum, this, new int[]{axis});
    }

    public Tensor max(int axis) {
        return g.OP(OPType.max, this, new int[]{axis});
    }

    public Tensor clamp(float min, float max) {
        return g.OP(OPType.clamp, this, min, max);
    }

    /**
     * 获取指定维度数据
     *
     * @param position int[dims,start,count]
     *                 <p>
     *                 dims: tensor 维度 0:number,1:channel,2:height,3:width
     *                 <p>
     *                 start: 指定维度开始脚标
     *                 <p>
     *                 count: 获取长度
     * @return
     */
    public Tensor get(int[] position) {
        return g.OP(OPType.get, this, position);
    }

    /**
     * 获取指定维度数据
     *
     * @param position int[dims,start,count]
     *                 <p>
     *                 dims: tensor 维度 0:number,1:channel,2:height,3:width
     *                 <p>
     *                 start: 指定维度开始脚标
     *                 <p>
     *                 count: 获取长度
     * @return
     */
    public Tensor set(Tensor target, int[] position) {
        return g.OP(OPType.set, this, target, position);
    }

    /**
     * 获取指定维度数据
     *
     * @param dim   tensor 维度 0:number,1:channel,2:height,3:width
     * @param start 指定维度开始脚标
     * @param count 获取长度
     */
    public Tensor get(int dim, int start, int count) {
        int[] position = new int[]{dim, start, count};
        return g.OP(OPType.get, this, position);
    }

    /**
     * 获取指定维度数据
     *
     * @param dim   tensor 维度 0:number,1:channel,2:height,3:width
     * @param start 指定维度开始脚标
     */
    public Tensor set(Tensor target, int dim, int start) {
        int[] position = new int[]{dim, start};
        return g.OP(OPType.set, this, target, position);
    }

    public void setGradByNumber(float[] data, int start, int count) {
        assert getShape()[0] >= (start + count - 1);
        System.arraycopy(data, 0, this.grad.getData(), start * getShape()[1] * getShape()[2] * getShape()[3], data.length);
    }

    public void setGradByChannel(float[] data, int start, int count) {
        assert getShape()[1] >= (start + count - 1);
        int size = getShape()[2] * getShape()[3];
        for (int n = 0; n < getShape()[0]; n++) {
            int startIndex = n * getShape()[1] * size + start * size;
            System.arraycopy(data, n * count * size, this.grad.getData(), startIndex, count * size);
        }
    }

    public void fill(float val, OPKernel kernel) {
        if (this.isHasGPU()) {
            kernel.fill_gpu(this, val);
        } else {
            MatrixUtils.val(this.getData(), val);
        }
    }

    public float[] getOnce() {
        if (this.once == null || this.once.length != getShape()[1] * getShape()[2] * getShape()[3]) {
            this.once = new float[getShape()[1] * getShape()[2] * getShape()[3]];
        }
        return once;
    }

    public Graph getG() {
        return g;
    }

    public void setG(Graph g) {
        this.g = g;
    }

    public void uniform(float min, float max) {
        for (int i = 0; i < this.getDataLength(); i++) {
            getData()[i] = RandomUtils.uniformFloat(min, max);
        }
        if (isHasGPU()) {
            hostToDevice();
        }
    }

    public boolean checkShape(Tensor y) {
        if (this.getShape()[0] == y.getShape()[0] && this.getShape()[1] == y.getShape()[1] && this.getShape()[2] == y.getShape()[2] && this.getShape()[3] == y.getShape()[3]) {
            return true;
        } else {
            return false;
        }
    }

    public SerializablePointer getShareGPU() {
        return shareGPU;
    }

    public void setShareGPU(SerializablePointer shareGPU) {
        this.shareGPU = shareGPU;
    }
    //	public void backward() {
    //		if(this.grad != null) {
    //			this.grad.fill(1.0f);
    //		}
    //		Graph.backward();
    //	}
}
