package com.omega.engine.ad.op.functions;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;

public class CosOP extends FunctionOP {
    public static final OPType opt = OPType.cos;
    /**
     *
     */
    private static final long serialVersionUID = -7252060328891832266L;
    public static CosOP op = null;

    public static CosOP getInstance() {
        if (op == null) {
            op = new CosOP();
            op.setOpType(opt);
        }
        return op;
    }

    @Override
    public Tensor forward(Tape tape) {
        // TODO Auto-generated method stub
        Tensor self = tape.getX();
        Tensor y = tape.getOutput();
        tape.getTensorOP().cos(self, y);
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
            tape.getTensorOP().sin(x, dy);
            tape.getTensorOP().mul(dy, -1, dy);
            tape.getTensorOP().mulPlus(delta, dy, x.getGrad());
        }
    }
}

