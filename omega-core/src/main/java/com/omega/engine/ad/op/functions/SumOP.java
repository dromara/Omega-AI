package com.omega.engine.ad.op.functions;

import com.omega.common.tensor.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;

public class SumOP extends FunctionOP {
    public static final OPType opt = OPType.sum;
    /**
     *
     */
    private static final long serialVersionUID = -3857343378511617891L;
    public static SumOP op = null;

    public static SumOP getInstance() {
        if (op == null) {
            op = new SumOP();
            op.setOpType(opt);
        }
        return op;
    }

    @Override
    public Tensor forward(Tape tape) {
        // TODO Auto-generated method stub
        Tensor self = tape.getX();
        Tensor y = tape.getOutput();
        tape.getTensorOP().sum(self, y, tape.getPosition()[0]);
        if (self.isRequiresGrad()) {
            y.setRequiresGrad(true);
        }
        return y;
    }

    /**
     *
     */
    @Override
    public void backward(Tensor delta, Tape tape) {
        // TODO Auto-generated method stub
        Tensor x = tape.getX();
        if (x.isRequiresGrad()) {
            tape.getTensorOP().broadcast(delta, x.getGrad(), tape.getPosition()[0]);
        }
    }
}

