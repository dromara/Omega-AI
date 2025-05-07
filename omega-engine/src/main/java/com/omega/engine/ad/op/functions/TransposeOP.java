package com.omega.engine.ad.op.functions;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;

public class TransposeOP extends FunctionOP {
    public static final OPType opt = OPType.transpose;
    /**
     *
     */
    private static final long serialVersionUID = -3857343378511617891L;
    public static TransposeOP op = null;

    public static TransposeOP getInstance() {
        if (op == null) {
            op = new TransposeOP();
            op.setOpType(opt);
        }
        return op;
    }

    @Override
    public Tensor forward(Tape tape) {
        // TODO Auto-generated method stub
        Tensor self = tape.getX();
        Tensor y = tape.getOutput();
        tape.getTensorOP().transpose(self, y);
        if (self.isRequiresGrad()) {
            y.setRequiresGrad(true);
        }
        return y;
    }

    /**
     * xt' = deltat
     */
    @Override
    public void backward(Tensor delta, Tape tape) {
        // TODO Auto-generated method stub
        Tensor x = tape.getX();
        if (x.isRequiresGrad()) {
            Tensor dy = tape.getTmp();
            tape.getTensorOP().transpose(delta, dy);
            tape.getTensorOP().mulPlus(dy, 1.0f, x.getGrad());
        }
    }
}

