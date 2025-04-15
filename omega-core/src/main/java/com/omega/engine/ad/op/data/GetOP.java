package com.omega.engine.ad.op.data;

import com.omega.common.tensor.Tensor;
import com.omega.utils.MatrixOperation;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OP;
import com.omega.engine.ad.op.OPType;

/**
 * 获取指定向量数据
 *
 * @author Administrator
 */
public class GetOP extends OP {
    public static final OPType opt = OPType.get;
    /**
     *
     */
    private static final long serialVersionUID = 7010180428917414516L;
    public static GetOP op = null;

    public static GetOP getInstance() {
        if (op == null) {
            op = new GetOP();
            op.setOpType(opt);
        }
        return op;
    }

    @Override
    public Tensor forward(Tape tape) {
        Tensor self = tape.getX();
        Tensor y = tape.getOutput();
        getByPosition(self, y, tape.getPosition(), tape);
        if (self.isRequiresGrad()) {
            y.setRequiresGrad(true);
        }
        return y;
    }

    @Override
    public void backward(Tensor delta, Tape tape) {
        // TODO Auto-generated method stub
        Tensor x = tape.getX();
        if (x.isRequiresGrad()) {
            addByPosition(x.getGrad(), delta, tape.getPosition(), tape);
        }
    }

    public void addByPosition(Tensor a, Tensor b, int[] position, Tape tape) {
        int dims = position[0];
        int start = position[1];
        int count = position[2];
        if (a.isHasGPU()) {
            switch (dims) {
                case 0:
                    tape.getTensorOP().op.add_number_gpu(a, b, start * count);
                    break;
                case 1:
                    tape.getTensorOP().op.add_channel_gpu(a, b, start * count);
                    break;
            }
        } else {
            int n = a.getNumber();
            int c = a.getChannel();
            int h = a.getHeight();
            int w = a.getWidth();
            MatrixOperation.add(a.data, b.data, n, c, h, w, position);
        }
    }

    public void getByPosition(Tensor org, Tensor target, int[] position, Tape tape) {
        int dims = position[0];
        int start = position[1];
        int count = position[2];
        switch (dims) {
            case 0:
                getByNumber(org, target, start, count, tape);
                break;
            case 1:
                getByChannel(org, target, start, count, tape);
                break;
            default:
                break;
        }
    }

    public void getByNumber(Tensor org, Tensor target, int start, int count, Tape tape) {
        assert org.getNumber() >= (start + count - 1);
        if (org.isHasGPU()) {
            tape.getTensorOP().op.copy_number_gpu(org, target, start * count, 0);
        } else {
            System.arraycopy(org.data, start * count * org.channel * org.height * org.width, target.data, 0, target.dataLength);
        }
    }

    public void getByChannel(Tensor org, Tensor target, int start, int count, Tape tape) {
        assert org.getChannel() >= (start + count - 1);
        if (org.isHasGPU()) {
            tape.getTensorOP().op.copy_channel_gpu(org, target, start, 0);
        } else {
            int size = org.height * org.width;
            for (int n = 0; n < org.number; n++) {
                int startIndex = n * org.channel * size + start * size;
                System.arraycopy(org.data, startIndex, target.data, n * count * size, count * size);
            }
        }
    }
}

