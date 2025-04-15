package com.omega.engine.ad.op.functions;

import com.omega.common.config.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;

public class SinOP extends FunctionOP {
    public static final OPType opt = OPType.sin;
    /**
     *
     */
    private static final long serialVersionUID = -7252060328891832266L;
    public static SinOP op = null;

    public static SinOP getInstance() {
        if (op == null) {
            op = new SinOP();
            op.setOpType(opt);
        }
        return op;
    }

    @Override
    public Tensor forward(Tape tape) {
        // TODO Auto-generated method stub
        Tensor self = tape.getX();
        Tensor y = tape.getOutput();
        tape.getTensorOP().sin(self, y);
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
            tape.getTensorOP().cos(x, dy);
            tape.getTensorOP().mulPlus(delta, dy, x.getGrad());
        }
    }
}

