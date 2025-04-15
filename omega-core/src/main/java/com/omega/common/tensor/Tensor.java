package com.omega.common.tensor;

import com.omega.utils.JsonUtils;
import com.omega.utils.MatrixOperation;
import com.omega.utils.MatrixUtils;
import com.omega.utils.RandomUtils;
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

public class Tensor implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 5844762745177624845L;
    public String id;
    public int number = 0;
    public int channel = 0;
    public int height = 0;
    public int width = 0;
    public int dataLength = 0;
    public int gpuLength = 0;
    public float[] data;
    public float[] once;
    public int onceSize;
    private Pointer gpuData;
    private SerializablePointer shareGPU;
    private boolean hasGPU = false;
    private boolean requiresGrad = false;
    private Tensor grad;
    private Graph g;
    private int[] orgShape;
    private Tensor tmp;
    private Tensor tmp_once;

    public Tensor(int number, int channel, int height, int width) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.data = new float[this.dataLength];
        this.orgShape = new int[]{number, channel, height, width};
    }

    public Tensor(int number, int channel, int height, int width, boolean hasGPU) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.data = new float[this.dataLength];
        this.orgShape = new int[]{number, channel, height, width};
        this.setHasGPU(hasGPU);
        if (hasGPU) {
            gpuData = CUDAMemoryManager.getPointer(dataLength);
            JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public Tensor(int number, int channel, int height, int width, boolean hasGPU, boolean onlyGPU) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.gpuLength = dataLength;
        if (!onlyGPU) {
            this.data = new float[this.dataLength];
        }
        this.orgShape = new int[]{number, channel, height, width};
        this.setHasGPU(hasGPU);
        if (hasGPU) {
            gpuData = CUDAMemoryManager.getPointer(dataLength);
            if (!onlyGPU) {
                JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
                JCuda.cudaDeviceSynchronize();
            } else {
                this.clearGPU();
            }
        }
    }

    public Tensor(int number, int channel, int height, int width, boolean hasGPU, Graph g) {
        this.g = g;
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.data = new float[this.dataLength];
        this.setHasGPU(hasGPU);
        if (hasGPU) {
            gpuData = CUDAMemoryManager.getPointer(dataLength);
            JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public Tensor(int number, int channel, int height, int width, float[] data) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.data = data;
        this.orgShape = new int[]{number, channel, height, width};
    }

    public Tensor(int number, int channel, int height, int width, float[] data, boolean hasGPU) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.data = data;
        this.orgShape = new int[]{number, channel, height, width};
        this.setHasGPU(hasGPU);
        if (hasGPU) {
            gpuData = CUDAMemoryManager.getPointer(dataLength);
            JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public Tensor(int number, int channel, int height, int width, float[] data, boolean hasGPU, boolean share) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.data = data;
        this.orgShape = new int[]{number, channel, height, width};
        this.setHasGPU(hasGPU);
        if (hasGPU) {
            setShareGPU(CUDAMemoryManager.getSharePointer(dataLength));
            JCuda.cudaMemcpy(getShareGPU(), Pointer.to(data), this.dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public Tensor(int number, int channel, int height, int width, float[] data, boolean hasGPU, Graph g) {
        this.g = g;
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.data = data;
        this.orgShape = new int[]{number, channel, height, width};
        this.setHasGPU(hasGPU);
        if (hasGPU) {
            gpuData = CUDAMemoryManager.getPointer(dataLength);
            JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public Tensor(int number, int channel, int height, int width, int val, boolean hasGPU) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.data = MatrixUtils.val(this.dataLength, val);
        this.orgShape = new int[]{number, channel, height, width};
        this.setHasGPU(hasGPU);
        if (hasGPU) {
            hostToDevice();
        }
    }

    public Tensor(int number, int channel, int height, int width, int val, boolean hasGPU, Graph g) {
        this.g = g;
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.data = MatrixUtils.val(this.dataLength, val);
        this.orgShape = new int[]{number, channel, height, width};
        this.setHasGPU(hasGPU);
        if (hasGPU) {
            hostToDevice();
        }
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

    public Tensor getTmp() {
        if (tmp == null) {
            tmp = new Tensor(number, channel, height, width, hasGPU);
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
        float[] dest = new float[dataLength];
        System.arraycopy(data, 0, dest, 0, dataLength);
        Tensor dis = new Tensor(number, channel, height, width, dest, hasGPU);
        return dis;
    }

    public Tensor copyGPU() {
        float[] dest = new float[dataLength];
        System.arraycopy(this.syncHost(), 0, dest, 0, dataLength);
        Tensor dis = new Tensor(number, channel, height, width, dest, hasGPU);
        return dis;
    }

    public void copy(Tensor tmp) {
        System.arraycopy(this.syncHost(), 0, tmp.data, 0, dataLength);
        if (tmp.hasGPU) {
            tmp.hostToDevice();
        }
    }

    public void copyGPU(Tensor tmp) {
        JCuda.cudaMemcpy(tmp.getGpuData(), gpuData, this.dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToDevice);
    }

    public Tensor createLike() {
        return new Tensor(number, channel, height, width, hasGPU);
    }

    public Tensor createLike(float value) {
        return new Tensor(number, channel, height, width, MatrixUtils.val(this.dataLength, value), hasGPU);
    }

    public void resize(int number, int channel, int height, int width) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.data = new float[this.dataLength];
        if (hasGPU) {
            if (gpuData != null) {
                CUDAMemoryManager.free(gpuData);
            }
            gpuData = CUDAMemoryManager.getPointer(dataLength);
        }
    }

    public void resize(int number, int channel, int height, int width, boolean onlyGPU) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.gpuLength = this.dataLength;
        if (!onlyGPU) {
            this.data = new float[this.dataLength];
        }
        if (hasGPU) {
            if (gpuData != null) {
                CUDAMemoryManager.free(gpuData);
            }
            gpuData = CUDAMemoryManager.getPointer(dataLength);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public void resize(int number, int channel, int height, int width, float[] data) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.data = data;
        if (hasGPU) {
            if (gpuData != null) {
                CUDAMemoryManager.free(gpuData);
            }
            gpuData = CUDAMemoryManager.getPointer(dataLength);
            JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public void copy(int n, float[] dest) {
        if (n < number) {
            System.arraycopy(data, n * channel * height * width, dest, 0, channel * height * width);
        } else {
            throw new RuntimeException("获取数据失败[下标超出长度].");
        }
    }

    public Tensor view(int number, int channel, int height, int width) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        return this;
    }

    public Tensor viewOrg(int number, int channel, int height, int width) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.orgShape[0] = number;
        this.orgShape[1] = channel;
        this.orgShape[2] = height;
        this.orgShape[3] = width;
        return this;
    }

    public Tensor view(int[] shape) {
        this.number = shape[0];
        this.channel = shape[1];
        this.height = shape[2];
        this.width = shape[3];
        return this;
    }

    public Tensor viewOrg() {
        this.number = orgShape[0];
        this.channel = orgShape[1];
        this.height = orgShape[2];
        this.width = orgShape[3];
        return this;
    }

    public int[] shape() {
        return new int[]{this.number, this.channel, this.height, this.width};
    }

    public void showShape() {
        System.out.println(JsonUtils.toJson(shape()));
    }

    public void showShape(String label) {
        System.out.println(label + ":" + JsonUtils.toJson(shape()));
    }

    public int getNumber() {
        return number;
    }

    public void setNumber(int number) {
        this.number = number;
    }

    public int getChannel() {
        return channel;
    }

    public void setChannel(int channel) {
        this.channel = channel;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int getDataLength() {
        return number * channel * height * width;
    }

    public void setDataLength(int dataLength) {
        this.dataLength = dataLength;
    }

    public int getOnceSize() {
        return channel * height * width;
    }

    public float[] getData() {
        return data;
    }

    public void setData(float[] data) {
        this.data = data;
        if (isHasGPU()) {
            this.hostToDevice();
        }
    }

    public void setData(int[] data) {
        for (int i = 0; i < data.length; i++) {
            this.data[i] = data[i] * 1.0f;
        }
        if (isHasGPU()) {
            this.hostToDevice();
        }
    }

    public void copyData(float[] data) {
        System.arraycopy(data, 0, this.data, 0, data.length);
        if (isHasGPU()) {
            this.hostToDevice();
        }
    }

    public float getByIndex(int n, int c, int h, int w) {
        return this.data[n * channel * height * width + c * height * width + h * width + w];
    }

    public float[] getByNumber(int n) {
        System.arraycopy(data, n * channel * height * width, getOnce(), 0, channel * height * width);
        return this.once;
    }

    public float[] getByOffset(int start, int len) {
        float[] tmp = new float[len];
        System.arraycopy(data, start, tmp, 0, len);
        return tmp;
    }

    public void setByNumber(int n, float[] x) {
        System.arraycopy(x, 0, data, n * channel * height * width, channel * height * width);
    }

    public void getByNumber(int n, float[] once) {
        if (once == null || once.length != channel * height * width) {
            once = new float[channel * height * width];
        }
        System.arraycopy(data, n * channel * height * width, once, 0, channel * height * width);
    }

    public float[] getByNumberAndChannel(int n, int c) {
        int start = n * channel * height * width + c * height * width;
        System.arraycopy(data, start, getOnce(), 0, height * width);
        return this.once;
    }

    public void setByNumberAndChannel(int n, int c, float[] x) {
        System.arraycopy(x, 0, data, n * channel * height * width + c * height * width, height * width);
    }

    public void getByNumberAndChannel(int n, int c, float[] once) {
        if (once == null || once.length != height * width) {
            once = new float[height * width];
        }
        int start = n * channel * height * width + c * height * width;
        System.arraycopy(data, start, once, 0, height * width);
    }

    public void clear() {
        for (int i = 0; i < this.dataLength; i++) {
            this.data[i] = 0;
        }
    }

    public void val_cpu(float val) {
        for (int i = 0; i < this.dataLength; i++) {
            this.data[i] = val;
        }
    }

    public void clear(int number, int channel, int height, int width) {
        this.number = number;
        this.channel = channel;
        this.height = height;
        this.width = width;
        this.dataLength = number * channel * height * width;
        this.data = new float[this.dataLength];
    }

    public Pointer getGpuData() {
        return gpuData;
    }

    public void setGpuData(Pointer gpuData) {
        this.gpuData = gpuData;
    }

    public float[] syncHost() {
        if (data == null || data.length != this.dataLength) {
            this.data = new float[this.dataLength];
        }
        JCuda.cudaMemcpy(Pointer.to(data), gpuData, this.dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        return data;
    }

    public void syncHost(float[] tmp) {
        JCuda.cudaMemcpy(Pointer.to(tmp), gpuData, this.dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
    }

    public void hostToDevice() {
        if (hasGPU) {
            if (gpuData == null) {
                gpuData = CUDAMemoryManager.getPointer(dataLength);
            }
            JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public void hostToDevice(float[] data) {
        if (hasGPU) {
            if (gpuData == null) {
                gpuData = CUDAMemoryManager.getPointer(dataLength);
            }
            JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * (long) Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaDeviceSynchronize();
        }
    }

    public void freeGPU() {
        if (gpuData != null) {
            JCuda.cudaFree(gpuData);
            gpuData = CUDAMemoryManager.getPointer(dataLength);
        }
    }

    public void showDM() {
        syncHost();
        System.out.println(JsonUtils.toJson(data));
    }

    public void showDMAndShape() {
        syncHost();
        System.out.println(JsonUtils.toJson(shape()) + ":" + JsonUtils.toJson(data));
    }

    public void showDM(String label) {
        syncHost();
        System.out.println(label + "[" + number + ":" + channel + ":" + height + ":" + width + "]" + JsonUtils.toJson(data));
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
        System.out.println(MatrixUtils.isZero(data));
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
        System.out.println(data[index]);
    }

    public void showDM(int index, String label) {
        syncHost();
        System.out.println(label + ":" + data[index]);
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
            this.grad = new Tensor(number, channel, height, width, this.hasGPU);
        }
        return grad;
    }

    public void setGrad(Tensor grad) {
        this.grad = grad;
    }

    public void setGrad(float[] grad) {
        if (this.grad == null) {
            this.grad = new Tensor(number, channel, height, width, grad, this.hasGPU);
        } else {
            this.grad.data = grad;
            this.grad.hostToDevice();
        }
    }

    public Tensor getGrad(float[] val) {
        if (this.grad == null) {
            this.grad = new Tensor(number, channel, height, width, val, this.hasGPU);
        }
        return grad;
    }

    public void setGrad(Tensor grad, int[] position, OPKernel kenel) {
        if (this.grad == null) {
            this.grad = new Tensor(number, channel, height, width, this.hasGPU);
        }
        int dims = position[0];
        int start = position[1];
        int count = position[2];
        switch (dims) {
            case 0:
                if (isHasGPU()) {
                    kenel.copy_number_gpu(this.grad, grad, start, 1);
                } else {
                    setGradByNumber(grad.data, start, count);
                }
                break;
            case 1:
                if (isHasGPU()) {
                    kenel.copy_channel_gpu(this.grad, grad, start, 1);
                } else {
                    setGradByChannel(grad.data, start, count);
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
        RandomUtils.gaussianRandom(data, 1.0f);
        if (isHasGPU()) {
            this.hostToDevice();
        }
    }

    public boolean isZero() {
        if (isHasGPU()) {
            return MatrixUtils.isZero(this.syncHost());
        }
        return MatrixUtils.isZero(data);
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
     * @param count 获取长度
     */
    public Tensor set(Tensor target, int dim, int start) {
        int[] position = new int[]{dim, start};
        return g.OP(OPType.set, this, target, position);
    }

    public void setGradByNumber(float[] data, int start, int count) {
        assert number >= (start + count - 1);
        System.arraycopy(data, 0, this.grad.data, start * channel * height * width, data.length);
    }

    public void setGradByChannel(float[] data, int start, int count) {
        assert channel >= (start + count - 1);
        int size = height * width;
        for (int n = 0; n < number; n++) {
            int startIndex = n * channel * size + start * size;
            System.arraycopy(data, n * count * size, this.grad.data, startIndex, count * size);
        }
    }

    public void fill(float val, OPKernel kernel) {
        if (this.isHasGPU()) {
            kernel.fill_gpu(this, val);
        } else {
            MatrixUtils.val(this.data, val);
        }
    }

    public float[] getOnce() {
        if (this.once == null || this.once.length != channel * height * width) {
            this.once = new float[channel * height * width];
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
        for (int i = 0; i < this.dataLength; i++) {
            data[i] = RandomUtils.uniformFloat(min, max);
        }
        if (isHasGPU()) {
            hostToDevice();
        }
    }

    public boolean checkShape(Tensor y) {
        if (this.number == y.number && this.channel == y.channel && this.height == y.height && this.width == y.width) {
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
