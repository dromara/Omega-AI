package com.omega.engine.ad.op.sign;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.SignOP;

public class ScalarSubOP extends SignOP {
    public static final OPType opt = OPType.subtraction;
    /**
     *
     */
    private static final long serialVersionUID = -3681016263960474439L;
    public static ScalarSubOP op = null;

    public static ScalarSubOP getInstance() {
        if (op == null) {
            op = new ScalarSubOP();
            op.setOpType(opt);
        }
        return op;
    }

    @Override
    public Tensor forward(Tape tape) {
        // TODO Auto-generated method stub
        Tensor self = tape.getX();
        Tensor y = tape.getOutput();
        tape.getTensorOP().sub(tape.getScalar(), self, y);
        if (self.isRequiresGrad()) {
            y.setRequiresGrad(true);
        }
        return y;
    }

    @Override
    public void backward(Tensor delta, Tape tape) {
        // TODO Auto-generated method stub
        Tensor x = tape.getX();
        if (x != null && x.isRequiresGrad()) {
            tape.getTensorOP().mulPlus(delta, -1.0f, x.getGrad());
        }
    }
}

