package com.omega.engine.ad.op.functions;

import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.tensor.Tensor;

public class MaxOP extends FunctionOP {
    public static final OPType opt = OPType.max;
    /**
     *
     */
    private static final long serialVersionUID = -3857343378511617891L;
    public static MaxOP op = null;

    public static MaxOP getInstance() {
        if (op == null) {
            op = new MaxOP();
            op.setOpType(opt);
        }
        return op;
    }

    @Override
    public Tensor forward(Tape tape) {
        // TODO Auto-generated method stub
        Tensor self = tape.getX();
        Tensor y = tape.getOutput();
        tape.getTensorOP().max(self, y, tape.getPosition()[0]);
        if (self.isRequiresGrad()) {
            y.setRequiresGrad(true);
        }
        return y;
    }

    /**
     * exp'(x) = exp(x)
     */
    @Override
    public void backward(Tensor delta, Tape tape) {
        // TODO Auto-generated method stub
        Tensor x = tape.getX();
        if (x.isRequiresGrad()) {
            tape.getTensorOP().max_backward(delta, x, x.getGrad(), tape.getPosition()[0]);
        }
    }
}

