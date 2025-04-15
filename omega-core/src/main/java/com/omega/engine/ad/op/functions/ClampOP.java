package com.omega.engine.ad.op.functions;

import com.omega.common.tensor.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;

public class ClampOP extends FunctionOP {
    public static final OPType opt = OPType.clamp;
    /**
     *
     */
    private static final long serialVersionUID = -6072156179108651118L;
    public static ClampOP op = null;

    public static ClampOP getInstance() {
        if (op == null) {
            op = new ClampOP();
            op.setOpType(opt);
        }
        return op;
    }

    @Override
    public Tensor forward(Tape tape) {
        // TODO Auto-generated method stub
        Tensor self = tape.getX();
        Tensor y = tape.getOutput();
        tape.getTensorOP().clamp(self, tape.getScalar(), tape.getConstant(), y);
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
            Tensor dy = tape.getTmp();
            tape.getTensorOP().clamp_back(x, tape.getScalar(), tape.getConstant(), dy);
            tape.getTensorOP().mulPlus(delta, dy, x.getGrad());
        }
    }
}

