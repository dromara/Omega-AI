package com.omega.engine.ad.op;

import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.ad.op.gpu.NormalizeKernel;
import com.omega.engine.ad.op.gpu.OPKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.tensor.Tensor;

import jcuda.driver.CUstream;
import jcuda.jcublas.cublasOperation;

public class TensorOP {
    public OPKernel op;
    private NormalizeKernel normalizeKernel;

    public TensorOP(CUDAManager cudaManager) {
        this.op = new OPKernel(cudaManager);
        this.normalizeKernel = new NormalizeKernel(cudaManager);
    }

    public void add(Tensor a, Tensor b, Tensor c) {
        if (c.isHasGPU()) {
            op.add_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.add(a.data, b.data);
        }
    }

    public void add(Tensor a, Tensor b, Tensor c, CUstream stream) {
        op.add_gpu(a, b, c, stream);
    }

    public void add(Tensor a, Tensor b, Tensor c, int axis) {
        if (c.isHasGPU()) {
            op.add_gpu(a, b, c, axis);
        } else {
            c.data = MatrixOperation.add(a.data, b.data);
        }
    }

    public void addAxis(Tensor a, Tensor b, Tensor c, int axis) {
        if (c.isHasGPU()) {
            op.add_axis_gpu(a, b, c, axis);
        } else {
            c.data = MatrixOperation.add(a.data, b.data);
        }
    }
    
    public void addAxis(Tensor a, Tensor b, Tensor c,int N, int C, int H, int W, int axis) {
        if (c.isHasGPU()) {
            op.add_axis_gpu(a, b, c, N, C, H, W, axis);
        } else {
            c.data = MatrixOperation.add(a.data, b.data);
        }
    }
    
    public void addAxisBack(Tensor dx, Tensor dy,int N, int C, int H, int W, int axis) {
        if (dx.isHasGPU()) {
            op.add_axis_back_gpu(dx, dy, N, C, H, W, axis);
        }
    }

    public void add(Tensor a, Tensor b, Tensor c, int offset, int N) {
        if (c.isHasGPU()) {
            op.add_gpu(a, b, c, offset, N);
        } else {
            c.data = MatrixOperation.add(a.data, b.data);
        }
    }

    public void add(Tensor a, Tensor b, Tensor c, int offsetA, int offsetB, int offsetC, int N) {
        if (c.isHasGPU()) {
            op.add_gpu(a, b, c, offsetA, offsetB, offsetC, N);
        } else {
            c.data = MatrixOperation.add(a.data, b.data);
        }
    }

    public void add(Tensor a, float b, Tensor c) {
        if (c.isHasGPU()) {
            op.add_scalar_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.add(a.data, b);
        }
    }

    public void sub(Tensor a, Tensor b, Tensor c) {
        int axis = getAxis(a, b);
        if (axis >= 0) {
            sub(a, b, c, axis);
        } else {
            if (c.isHasGPU()) {
                op.sub_gpu(a, b, c);
            } else {
                c.data = MatrixOperation.subtraction(a.data, b.data);
            }
        }
    }

    public int getAxis(Tensor a, Tensor b) {
        if (a.getDataLength() == b.getDataLength()) {
            return -1;
        }
        return 0;
    }

    public void sub(Tensor a, Tensor b, Tensor c, int axis) {
        if (c.isHasGPU()) {
            op.sub_gpu(a, b, c, axis);
        } else {
            c.data = MatrixOperation.subtraction(a.data, b.data, axis);
        }
    }

    public void sub(Tensor a, Tensor b, Tensor c, int offset, int N) {
        if (c.isHasGPU()) {
            op.sub_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.subtraction(a.data, b.data);
        }
    }

    public void sub(Tensor a, float b, Tensor c) {
        if (c.isHasGPU()) {
            op.sub_scalar_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.subtraction(a.data, b);
        }
    }

    public void sub(float a, Tensor b, Tensor c) {
        if (c.isHasGPU()) {
            op.scalar_sub_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.subtraction(a, b.data);
        }
    }

    public void sub(float a, Tensor b, Tensor c, int offset, int N) {
        if (c.isHasGPU()) {
            op.scalar_sub_gpu(a, b, c, offset, N);
        } else {
            c.data = MatrixOperation.subtraction(a, b.data);
        }
    }

    public void mul(Tensor a, Tensor b, Tensor c) {
        if (c.isHasGPU()) {
            op.mul_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.multiplication(a.data, b.data);
        }
    }
    
    public void bool(Tensor a, Tensor b, Tensor c, float val) {
        if (c.isHasGPU()) {
            op.bool_gpu(a, b, c, val);
        } else {
            c.data = MatrixOperation.bool(a.data, b.data, val);
        }
    }
    
    public void mask(Tensor a, Tensor b, Tensor c, float val) {
        if (c.isHasGPU()) {
            op.mask_gpu(a, b, c, val);
        }else {
        	
        }
    }
    
    public void mask(Tensor a, Tensor b, Tensor c, float val, int dataLen, int onceSize) {
        if (c.isHasGPU()) {
            op.mask_gpu(a, b, c, val, dataLen, onceSize);
        }else {
        	
        }
    }
    
    public void mul(Tensor a, Tensor b, Tensor c, int offset, int N) {
        if (c.isHasGPU()) {
            op.mul_gpu(a, b, c, offset, N);
        } else {
            c.data = MatrixOperation.multiplication(a.data, b.data);
        }
    }
    
    public void mul(Tensor a, Tensor b, Tensor c, int N, int C, int H, int W, int axis) {
        if (c.isHasGPU()) {
            op.mul_gpu(a, b, c, N, C, H, W, axis);
        }
    }
    
    public void mul_left_back(Tensor b, Tensor delta, Tensor da, int N, int C, int H, int W, int axis) {
        if (da.isHasGPU()) {
            op.mul_back_left_gpu(b, delta, da, N, C, H, W, axis);
        }
    }
    
    public void mul_right_back(Tensor a, Tensor delta, Tensor db, int N, int C, int H, int W, int axis) {
        if (db.isHasGPU()) {
            op.mul_back_right_gpu(a, delta, db, N, C, H, W, axis);
        }
    }

    public void mul(Tensor a, Tensor b, Tensor c, int offsetA, int offsetB, int offsetY, int N) {
        if (c.isHasGPU()) {
            op.mul_gpu(a, b, c, offsetA, offsetB, offsetY, N);
        } else {
            c.data = MatrixOperation.multiplication(a.data, b.data);
        }
    }

    public void mul(Tensor a, float b, Tensor c) {
        if (c.isHasGPU()) {
            op.mul_scalar_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.multiplication(a.data, b);
        }
    }

    public void mulPlus(Tensor a, Tensor b, Tensor c) {
        if (c.isHasGPU()) {
            op.mul_plus_gpu(a, b, c);
        } else {
            MatrixOperation.plus(c.data, MatrixOperation.multiplication(a.data, b.data));
        }
    }

    public void mulPlus(Tensor a, float b, Tensor c) {
        int axis = getAxis(a, c);
        if (axis >= 0) {
            mulPlus(a, b, c, axis);
        } else {
            if (c.isHasGPU()) {
                op.mul_plus_scalar_gpu(a, b, c);
            } else {
                MatrixOperation.plus(c.data, MatrixOperation.multiplication(a.data, b));
            }
        }
    }

    public void mulPlus(Tensor a, float b, Tensor c, int axis) {
        if (c.isHasGPU()) {
            op.mul_plus_scalar_gpu(a, b, c, axis);
        } else {
            MatrixOperation.plus(c.data, MatrixOperation.multiplication(a.data, b), axis);
        }
    }

    public void div(Tensor a, Tensor b, Tensor c) {
        int axis = getAxis(a, b);
        if (axis >= 0) {
            div(a, b, c, axis);
        } else {
            if (c.isHasGPU()) {
                op.div_gpu(a, b, c);
            } else {
                c.data = MatrixOperation.division(a.data, b.data);
            }
        }
    }

    public void div(Tensor a, Tensor b, Tensor c, CUstream stream) {
        op.div_gpu(a, b, c, stream);
    }

    public void div(Tensor a, Tensor b, Tensor c, int axis) {
        if (c.isHasGPU()) {
            op.div_gpu(a, b, c, axis);
        } else {
            c.data = MatrixOperation.division(a.data, b.data, axis);
        }
    }

    public void div(Tensor a, float b, Tensor c) {
        if (c.isHasGPU()) {
            op.div_scalar_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.division(a.data, b);
        }
    }

    public void div(Tensor a, float b, Tensor c, CUstream stream) {
        op.div_scalar_gpu(a, b, c, stream);
    }

    public void div(float a, Tensor b, Tensor c) {
        if (c.isHasGPU()) {
            op.scalar_div_gpu(b, a, c);
        } else {
            c.data = MatrixOperation.division(a, b.data);
        }
    }

    public void divPlus(Tensor a, Tensor b, Tensor c) {
        int axis = getAxis(a, b);
        if (axis >= 0) {
            divPlus(a, b, c, axis);
        } else {
            if (c.isHasGPU()) {
                op.div_plus_gpu(a, b, c);
            } else {
                MatrixOperation.plus(c.data, MatrixOperation.division(a.data, b.data));
            }
        }
    }

    public void divPlus(Tensor a, Tensor b, Tensor c, int axis) {
        if (c.isHasGPU()) {
            op.div_plus_gpu(a, b, c, axis);
        } else {
            MatrixOperation.plus(c.data, MatrixOperation.division(a.data, b.data, axis));
        }
    }

    public void divPlus(Tensor a, float b, Tensor c) {
        if (c.isHasGPU()) {
            op.div_plus_scalar_gpu(a, b, c);
        } else {
            MatrixOperation.plus(c.data, MatrixOperation.division(a.data, b));
        }
    }

    public void exp(Tensor a, Tensor b) {
        if (b.isHasGPU()) {
            op.exp_gpu(a, b);
        } else {
            b.data = MatrixOperation.exp(a.data);
        }
    }

    public void transpose(Tensor a, Tensor b) {
        if (b.isHasGPU()) {
            op.transpose_gpu(a, b);
        } else {
            //			b.data = MatrixOperation.exp(a.data);
        }
    }

    public void sum(Tensor a, Tensor b, int axis) {
        if (b.isHasGPU()) {
            op.sum_gpu(a, b, axis);
        } else {
            b.data = MatrixOperation.sum(a.data, a.number, a.channel, a.height, a.width, axis);
        }
    }

    public void sum_pow(Tensor a, Tensor b, double p, int axis) {
        if (b.isHasGPU()) {
            op.sum_pow_gpu(a, b, p, axis);
        }
    }

    public void max(Tensor a, Tensor b, int axis) {
        if (b.isHasGPU()) {
            op.max_gpu(a, b, axis);
        } else {
            b.data = MatrixOperation.max(a.data, a.number, a.channel, a.height, a.width, axis);
        }
    }

    public void max_backward(Tensor d, Tensor a, Tensor b, int axis) {
        if (b.isHasGPU()) {
            op.max_backward_gpu(d, a, b, axis);
        } else {
            b.data = MatrixOperation.max(a.data, a.number, a.channel, a.height, a.width, axis);
        }
    }

    public void log(Tensor a, Tensor b) {
        if (b.isHasGPU()) {
            op.log_gpu(a, b);
        } else {
            b.data = MatrixOperation.log(a.data);
        }
    }

    public void pow(Tensor a, float b, Tensor c) {
        if (c.isHasGPU()) {
            op.pow_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.pow(a.data, b);
        }
    }

    public void sqrt(Tensor a, Tensor c) {
        if (c.isHasGPU()) {
            op.sqrt_gpu(a, c);
        } else {
            c.data = MatrixOperation.sqrt(a.data);
        }
    }

    public void sin(Tensor a, Tensor c) {
        if (c.isHasGPU()) {
            op.sin_gpu(a, c);
        } else {
            c.data = MatrixOperation.sin(a.data);
        }
    }

    public void cos(Tensor a, Tensor c) {
        if (c.isHasGPU()) {
            op.cos_gpu(a, c);
        } else {
            c.data = MatrixOperation.cos(a.data);
        }
    }

    public void tan(Tensor a, Tensor c) {
        if (c.isHasGPU()) {
            op.tan_gpu(a, c);
        } else {
            c.data = MatrixOperation.tan(a.data);
        }
    }

    public void tan_back(Tensor a, Tensor c) {
        if (c.isHasGPU()) {
            op.tan_back_gpu(a, c);
        } else {
            c.data = MatrixOperation.tan_back(a.data);
        }
    }

    public void atan(Tensor a, Tensor c) {
        if (c.isHasGPU()) {
            op.atan_gpu(a, c);
        } else {
            c.data = MatrixOperation.atan(a.data);
        }
    }

    public void atan_back(Tensor a, Tensor c) {
        if (c.isHasGPU()) {
            op.atan_back_gpu(a, c);
        } else {
            c.data = MatrixOperation.atan_back(a.data);
        }
    }

    public void broadcast(Tensor a, Tensor c, int axis) {
        if (c.isHasGPU()) {
            op.broadcast_plus_gpu(a, c, axis);
        } else {
            MatrixOperation.broadcast_plus(a.data, c.data, c.number, c.channel, c.height, c.width, axis);
        }
    }

    public void broadcast_row(Tensor a, Tensor c) {
        if (c.isHasGPU()) {
            op.broadcast_row_plus_gpu(a, c);
        }
    }

    public void clamp(Tensor a, float b1, float b2, Tensor c) {
        if (c.isHasGPU()) {
            op.clamp_gpu(a, b1, b2, c);
        } else {
            c.data = MatrixOperation.clamp(a.data, b1, b2);
        }
    }

    public void clamp_back(Tensor a, float b1, float b2, Tensor c) {
        if (c.isHasGPU()) {
            op.clamp_back_gpu(a, b1, b2, c);
        } else {
            c.data = MatrixOperation.clamp_back(a.data, b1, b2);
        }
    }

    public void maximum(Tensor a, Tensor b, Tensor c) {
        if (c.isHasGPU()) {
            op.maximum_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.maximum(a.data, b.data);
        }
    }

    public void minimum(Tensor a, Tensor b, Tensor c) {
        if (c.isHasGPU()) {
            op.minimum_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.minimum(a.data, b.data);
        }
    }

    public void maximum_back(Tensor a, Tensor b, Tensor c) {
        if (c.isHasGPU()) {
            op.maximum_back_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.maximum_back(a.data, b.data);
        }
    }

    public void minimum_back(Tensor a, Tensor b, Tensor c) {
        if (c.isHasGPU()) {
            op.minimum_back_gpu(a, b, c);
        } else {
            c.data = MatrixOperation.minimum_back(a.data, b.data);
        }
    }

    public void mean(Tensor a, int dim, Tensor c) {
        if (c.isHasGPU()) {
            op.mean_gpu(a, dim, c);
        } else {
            c.data = MatrixOperation.mean(a.data, a.number, a.channel, a.height, a.width, dim);
        }
    }

    public void main(String[] args) {
        int B = 2;
        int C = 4;
        int H = 3;
        int W = 3;
        Tensor x = new Tensor(B, C, H, W, MatrixUtils.order(B * C * H * W, 1.0f, 1.0f), true);
        Tensor r = new Tensor(1, 1, 1, 1, true);
        CUDAManager cudaManager = new CUDAManager(0);
        TensorOP op = new TensorOP(cudaManager);
        op.mean(x, 0, r);
        x.showDM();
        r.showDM();
        System.out.println(r.syncHost()[0] / C / H / W);
    }

    /**
     * [M,N] dot [N,K]
     *
     * @param a
     * @param b
     * @param c
     * @param A_OP
     * @param b_OP
     */
    public void dot(Tensor a, Tensor b, Tensor c) {
        if (c.isHasGPU()) {
            /**
             * m = M,n = K,k = N
             * batch, oWidth, width

             */
            //			System.out.println(JsonUtils.toJson(a.shape()));
            //			System.out.println(JsonUtils.toJson(b.shape()));
            //			a.showDM();
            //			b.showDM();
            int k = b.number;
            if (b.number == 1) {
                k = b.height;
            }
            GPUOP.getInstance().multiplyFloat(a.number, b.width, k, a.getGpuData(), b.getGpuData(), c.getGpuData(), cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
            //			c.showDM();
            //			System.out.println("----------------------");
        } else {
            //			c.data = MatrixOperation.dot(a.data, b.data);
        }
    }

    /**
     * diff = delta * weightT
     * <p>
     * this.number, this.width, this.oWidth
     *
     * @param a
     * @param b
     * @param c
     */
    public void dotDX(Tensor a, Tensor b, Tensor c) {
        int k = b.number;
        if (b.number == 1) {
            k = b.height;
        }
        GPUOP.getInstance().multiplyFloat(a.number, k, b.width, a.getGpuData(), b.getGpuData(), c.getGpuData(), cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 1.0f);
    }

    /**
     * deltaW = inputT * delta
     * <p>
     * this.width, this.oWidth, this.number
     *
     * @param a
     * @param b
     * @param c
     */
    public void dotDW(Tensor a, Tensor b, Tensor c) {
        GPUOP.getInstance().multiplyFloat(a.width, b.width, a.number, a.getGpuData(), b.getGpuData(), c.getGpuData(), cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 1.0f);
    }

    public void permute(Tensor a, Tensor b, int[] permutes) {
        if (a.isHasGPU()) {
            op.permute_gpu(a, b, permutes);
        } else {
            //			c.data = MatrixOperation.add(a.data, b.data);
        }
    }

    public void permuteAdd(Tensor a, Tensor b, int[] permutes) {
        if (a.isHasGPU()) {
            op.permute_add_gpu(a, b, permutes);
        } else {
            //			c.data = MatrixOperation.add(a.data, b.data);
        }
    }
    
    public void permute(Tensor a, Tensor b, int[] xShape, int[] yShape, int[] permutes) {
        if (a.isHasGPU()) {
            op.permute_gpu(a, b, xShape, yShape, permutes);
        } else {
            //			c.data = MatrixOperation.add(a.data, b.data);
        }
    }
    
    public void permute(Tensor a, Tensor b, int[] xShape, int[] yShape, int[] permutes, int len) {
        if (a.isHasGPU()) {
            op.permute_gpu(a, b, xShape, yShape, permutes, len);
        } else {
            //			c.data = MatrixOperation.add(a.data, b.data);
        }
    }

    public void expand(Tensor a, Tensor b, int num) {
        if (a.isHasGPU()) {
            op.expand_gpu(a, b, num);
        } else {
            //			c.data = MatrixOperation.add(a.data, b.data);
        }
    }

    public void cat(Tensor a, Tensor b, Tensor c) {
        if (a.isHasGPU()) {
            op.cat_gpu(a, b, c);
        } else {
            //			c.data = MatrixOperation.add(a.data, b.data);
        }
    }

    public void cat_back(Tensor c, Tensor a, Tensor b) {
        if (a.isHasGPU()) {
            op.cat_back_gpu(c, a, b);
        } else {
            //			c.data = MatrixOperation.add(a.data, b.data);
        }
    }
    
    public void cat_width(Tensor a, Tensor b, Tensor c) {
        if (a.isHasGPU()) {
            op.cat_width_gpu(a, b, c);
        } else {
            //			c.data = MatrixOperation.add(a.data, b.data);
        }
    }

    public void cat_width_back(Tensor c, Tensor a, Tensor b) {
        if (a.isHasGPU()) {
            op.cat_width_back_gpu(c, a, b);
        } else {
            //			c.data = MatrixOperation.add(a.data, b.data);
        }
    }
    
    public void onehot(Tensor a, Tensor b) {
        if (a.isHasGPU()) {
            op.one_hot(a, b);
        } else {
            //			c.data = MatrixOperation.add(a.data, b.data);
        }
    }

    public void mean2Dim(Tensor a, Tensor b) {
        if (a.isHasGPU()) {
            op.mean_2dim_gpu(a, b);
        } else {
            //			c.data = MatrixOperation.add(a.data, b.data);
        }
    }

    public void mean2DimBack(Tensor dy, Tensor dx) {
        if (dx.isHasGPU()) {
            op.mean_2dim_back_gpu(dy, dx);
        } else {
            //			c.data = MatrixOperation.add(a.data, b.data);
        }
    }

    public void copyGPU(Tensor a, Tensor b) {
        op.copy_gpu(a, b);
    }
    
    public Tensor copyTensorGPU(Tensor a, Tensor b) {
    	if(b == null) {
    		b = a.createLike();
    	}
        op.copy_gpu(a, b);
        return b;
    }

    public void normalize(Tensor x, Tensor y, int dim) {
        int N = x.number;
        int C = x.channel;
        int H = x.height;
        int W = x.width;
        if (dim == 0) {
            x.view(N * C * H * W, 1, 1, 1);
            y.view(N * C * H * W, 1, 1, 1);
        } else if (dim == 1) {
            x.view(N, 1, 1, C * H * W);
            y.view(N, 1, 1, C * H * W);
        } else if (dim == 2) {
            x.view(N * C, 1, 1, H * W);
            y.view(N * C, 1, 1, H * W);
        } else if (dim == 3) {
            x.view(N * C * H, 1, 1, W);
            y.view(N * C * H, 1, 1, W);
        } else {
            throw new RuntimeException("dim must be 0 to 3");
        }
        normalizeKernel.l2norm(x, y);
        x.view(N, C, H, W);
        y.view(N, C, H, W);
    }
    
    /**
     * cp = 0 [org->target]
     * cp = 1 [target->org]
     * @param org
     * @param target
     * @param position[dims, start, count]
     * @param cp
     */
    public void getByPosition(Tensor org, Tensor target, int[] position, int cp) {
        int dims = position[0];
        int start = position[1];
        int count = position[2];
        if(cp == 0) {
        	 switch (dims) {
	             case 0:
	                 getByNumber(org, target, start, count);
	                 break;
	             case 1:
	                 getByChannel(org, target, start, count);
	                 break;
	             default:
	                 break;
	         }
        }else {
        	switch (dims) {
	            case 0:
	                getByNumber_back(org, target, start, count);
	                break;
	            case 1:
	                getByChannel_back(org, target, start, count);
	                break;
	            default:
	                break;
	        }
        }
       
    }
    
    public void getByNumber(Tensor org, Tensor target, int start, int count) {
        assert org.getNumber() >= (start + count - 1);
        if (org.isHasGPU()) {
            this.op.copy_number_gpu(org, target, start * count, 0);
        } else {
            System.arraycopy(org.data, start * count * org.channel * org.height * org.width, target.data, 0, target.dataLength);
        }
    }

    public void getByChannel(Tensor org, Tensor target, int start, int count) {
        assert org.getChannel() >= (start + count - 1);
        if (org.isHasGPU()) {
            this.op.copy_channel_gpu(org, target, start, 0);
        } else {
            int size = org.height * org.width;
            for (int n = 0; n < org.number; n++) {
                int startIndex = n * org.channel * size + start * size;
                System.arraycopy(org.data, startIndex, target.data, n * count * size, count * size);
            }
        }
    }
    
    public void getByChannel(Tensor org, Tensor target,int[] shape, int start, int count) {
        if (org.isHasGPU()) {
            this.op.copy_channel_gpu(org, target, shape, start, 0);
        }
    }
    
    public void getByChannel(Tensor org, Tensor target,int[] shape, int start) {
        if (org.isHasGPU()) {
            this.op.copy_channel_gpu(org, target, shape, start, 0);
        }
    }
    
    public void setByChannel(Tensor org, Tensor target,int[] shape, int start) {
        if (org.isHasGPU()) {
            this.op.copy_channel_gpu_set(org, target, shape, start, 1);
        }
    }
    
    public void addByChannel(Tensor org, Tensor target,int[] shape, int start) {
        if (org.isHasGPU()) {
            this.op.add_channel_gpu(org, target, shape, start);
        }
    }
    
    public void getByNumber_back(Tensor org, Tensor target, int start, int count) {
        assert org.getNumber() >= (start + count - 1);
        if (org.isHasGPU()) {
            this.op.copy_number_gpu(org, target, start * count, 1);
        } else {
            System.arraycopy(org.data, start * count * org.channel * org.height * org.width, target.data, 0, target.dataLength);
        }
    }

    public void getByChannel_back(Tensor org, Tensor target, int start, int count) {
        assert org.getChannel() >= (start + count - 1);
        if (org.isHasGPU()) {
            this.op.copy_channel_gpu(org, target, start, 1);
        } else {
            int size = org.height * org.width;
            for (int n = 0; n < org.number; n++) {
                int startIndex = n * org.channel * size + start * size;
                System.arraycopy(org.data, startIndex, target.data, n * count * size, count * size);
            }
        }
    }
    
    public void getByChannel_back(Tensor org, Tensor target, int[] shape, int start, int count) {
        if (org.isHasGPU()) {
            this.op.copy_channel_gpu(org, target, shape, start, 1);
        }
    }
    
    public void getByChannel_back(Tensor org, Tensor target, int[] shape, int start) {
        if (org.isHasGPU()) {
            this.op.copy_channel_gpu(org, target, shape, start, 1);
        }
    }
    
    public void abs(Tensor x,Tensor y) {
    	op.abs_gpu(x, y);
    }
    
    public void abs_backward(Tensor x,Tensor dx,Tensor dy) {
    	op.abs_backward_gpu(x, dx, dy);
    }
    
    public void update_ema(Tensor ema,Tensor model,float decay) {
    	op.update_ema_gpu(ema, model, decay);
    }
    
}

