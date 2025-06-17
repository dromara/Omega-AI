package com.omega.engine.ad.op.data;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OP;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.tensor.Tensor;

/**
 * 获取指定向量数据
 *
 * @author Administrator
 */
public class SetOP extends OP {
    public static final OPType opt = OPType.set;
    /**
     *
     */
    private static final long serialVersionUID = 7010180428917414516L;
    public static SetOP op = null;

    public static SetOP getInstance() {
        if (op == null) {
            op = new SetOP();
            op.setOpType(opt);
        }
        return op;
    }

    @Override
    public Tensor forward(Tape tape) {
        Tensor self = tape.getX();
        Tensor y = tape.getY();
        setByPosition(self, y, tape.getPosition(), tape);
        if (self.isRequiresGrad()) {
            y.setRequiresGrad(true);
        }
        return y;
    }

    @Override
    public void backward(Tensor delta, Tape tape) {
        // TODO Auto-generated method stub
        Tensor y = tape.getY();
        if (y.isRequiresGrad()) {
            addByPosition(y.getGrad(), delta, tape.getPosition(), tape);
        }
    }

    public void addByPosition(Tensor a, Tensor b, int[] position, Tape tape) {
        int dims = position[0];
        int start = position[1];
        if (a.isHasGPU()) {
            switch (dims) {
                case 0:
                    tape.getTensorOP().op.axpy_gpu(b, a, start * a.getShape()[1] * a.getShape()[2] * a.getShape()[3], 0);
                    break;
            }
        } else {
            int n = a.getNumber();
            int c = a.getChannel();
            int h = a.getHeight();
            int w = a.getWidth();
            MatrixOperation.add(b.getData(), a.getData(), n, c, h, w, position);
        }
    }

    public void setByPosition(Tensor org, Tensor target, int[] position, Tape tape) {
        int dims = position[0];
        int start = position[1];
        switch (dims) {
            case 0:
                setByNumber(org, target, start, tape);
                break;
            case 1:
                break;
            default:
                break;
        }
    }

    public void setByNumber(Tensor org, Tensor target, int start, Tape tape) {
        assert org.getNumber() >= (start - 1);
        if (org.isHasGPU()) {
            tape.getTensorOP().op.copy_gpu(target, org, 0, start * target.getShape()[1] * target.getShape()[2] * target.getShape()[3]);
        } else {
            System.arraycopy(target.getData(), 0, org.getData(), start * target.getShape()[1] * target.getShape()[2] * target.getShape()[3], target.getDataLength());
        }
    }
}

