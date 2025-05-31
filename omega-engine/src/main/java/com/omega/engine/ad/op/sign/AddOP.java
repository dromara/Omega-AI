package com.omega.engine.ad.op.sign;

import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.SignOP;
import com.omega.engine.tensor.Tensor;

/**
 * 加法操作
 *
 * @author Administrator
 */
public class AddOP extends SignOP {
    public static final OPType opt = OPType.add;
    /**
     *
     */
    private static final long serialVersionUID = -6030727723343775529L;
    public static AddOP op = null;

    public static AddOP getInstance() {
        if (op == null) {
            op = new AddOP();
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
        if (tape.getPosition() != null) {
        } else {
            if (other != null) {
                tape.getTensorOP().add(self, other, y);
            } else {
                tape.getTensorOP().add(self, tape.getScalar(), y);
            }
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
            tape.getTensorOP().mulPlus(delta, 1.0f, y.getGrad());
        }
    }
}

