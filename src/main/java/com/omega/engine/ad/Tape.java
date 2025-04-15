package com.omega.engine.ad;

import com.omega.common.config.Tensor;
import com.omega.engine.ad.op.OP;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.TensorOP;

import java.io.Serializable;

/**
 * 计算图节点
 *
 * @author Administrator
 */
public class Tape implements Serializable {
    /**
     *
     */
    private static final long serialVersionUID = 9147342370353517536L;
    private Tensor x;
    private Tensor y;
    private Tensor output;
    private int[] position;
    private OP op;
    private float scalar;
    private float constant;
    private Tensor tmp;
    private boolean sub = false;
    private TensorOP tensorOP;

    public Tape(OP op, Tensor self, Tensor other, float scalar, float constant, int[] position, Graph g) {
        this.tensorOP = g.getTensorOP();
        this.setX(self);
        this.setY(other);
        if (position != null && !op.getOpType().equals(OPType.set)) {
            int dims = position[0];
            if (!op.getOpType().equals(OPType.sum) && !op.getOpType().equals(OPType.max)) {
                int count = position[2];
                switch (dims) {
                    case 0:
                        setOutput(new Tensor(count, self.channel, self.height, self.width, self.isHasGPU(), g));
                        break;
                    case 1:
                        setOutput(new Tensor(self.number, count, self.height, self.width, self.isHasGPU(), g));
                        break;
                }
            } else {
                switch (dims) {
                    case 0:
                        setOutput(new Tensor(1, 1, 1, 1, self.isHasGPU(), g));
                        break;
                    case 1:
                        setOutput(new Tensor(self.number, 1, 1, 1, self.isHasGPU(), g));
                        break;
                }
            }
        } else {
            if (op.getOpType().equals(OPType.dot)) {
                setOutput(new Tensor(self.number, self.channel, self.height, other.width, self.isHasGPU(), g));
            } else if (op.getOpType().equals(OPType.set)) {
                this.output = self;
            } else if (op.getOpType().equals(OPType.transpose)) {
                setOutput(new Tensor(self.width, self.channel, self.height, self.number, self.isHasGPU(), g));
            } else {
                setOutput(new Tensor(self.number, self.channel, self.height, self.width, self.isHasGPU(), g));
            }
        }
        this.setOp(op);
        this.scalar = scalar;
        this.constant = constant;
        this.setPosition(position);
    }

    public OP getOp() {
        return op;
    }

    public void setOp(OP op) {
        this.op = op;
    }

    public void zeroGrad() {
        if (getX().isRequiresGrad()) {
            getX().zeroGrad(tensorOP.op);
        }
        if (getY() != null && getY().isRequiresGrad()) {
            getY().zeroGrad(tensorOP.op);
        }
        if (getOutput().isRequiresGrad()) {
            getOutput().zeroGrad(tensorOP.op);
        }
    }

    public Tensor forward() {
        return this.op.forward(this);
    }

    public void backward(Tensor delta) {
        op.backward(delta, this);
        //		if(this.getPosition() != null) {
        //			GetOP getOp = (GetOP) op;
        //			getOp.backward(delta, this);
        //		}else {
        //			op.backward(delta, this);
        //		}
    }

    public void backward() {
        this.backward(getOutput().getGrad());
    }

    public float getScalar() {
        return scalar;
    }

    public void setScalar(float scalar) {
        this.scalar = scalar;
    }

    public int[] getPosition() {
        return position;
    }

    public void setPosition(int[] position) {
        this.position = position;
    }

    public Tensor getX() {
        return x;
    }

    public void setX(Tensor x) {
        this.x = x;
    }

    public Tensor getY() {
        return y;
    }

    public void setY(Tensor y) {
        this.y = y;
    }

    public Tensor getOutput() {
        return output;
    }

    public void setOutput(Tensor output) {
        this.output = output;
    }

    public Tensor getTmp() {
        if (tmp == null) {
            this.tmp = new Tensor(this.x.number, this.x.channel, this.x.height, this.x.width, this.x.isHasGPU());
        }
        return tmp;
    }

    public void setTmp(Tensor tmp) {
        this.tmp = tmp;
    }

    public boolean isSub() {
        return sub;
    }

    public void setSub(boolean sub) {
        this.sub = sub;
    }

    public float getConstant() {
        return constant;
    }

    public void setConstant(float constant) {
        this.constant = constant;
    }

    public TensorOP getTensorOP() {
        return tensorOP;
    }
}

