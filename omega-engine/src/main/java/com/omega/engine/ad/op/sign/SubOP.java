package com.omega.engine.ad.op.sign;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.SignOP;

public class SubOP extends SignOP {
    public static final OPType opt = OPType.subtraction;
    /**
     *
     */
    private static final long serialVersionUID = -3681016263960474439L;
    public static SubOP op = null;

    public static SubOP getInstance() {
        if (op == null) {
            op = new SubOP();
            op.setOpType(opt);
        }
        return op;
    }

    @Override
    public Tensor forward(Tape tape) {
        // TODO Auto-generated method stub
        Tensor self = tape.getX();
        Tensor other = tape.getY();
        Tensor y = tape.getOutput();
        if (other != null) {
            tape.getTensorOP().sub(self, other, y);
        } else {
            tape.getTensorOP().sub(self, tape.getScalar(), y);
        }
        if (self.isRequiresGrad() || other.isRequiresGrad()) {
            y.setRequiresGrad(true);
        }
        return y;
    }

    @Override
    public void backward(Tensor delta, Tape tape) {
        // TODO Auto-generated method stub
        Tensor x = tape.getX();
        Tensor y = tape.getY();
        if (x.isRequiresGrad()) {
            tape.getTensorOP().mulPlus(delta, 1.0f, x.getGrad());
        }
        if (y != null && y.isRequiresGrad()) {
            tape.getTensorOP().mulPlus(delta, -1.0f, y.getGrad());
        }
    }
}

