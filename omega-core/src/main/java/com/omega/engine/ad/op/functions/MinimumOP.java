package com.omega.engine.ad.op.functions;

import com.omega.common.tensor.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;

/**
 * MinimumOP
 *
 * @author Administrator
 */
public class MinimumOP extends FunctionOP {
    public static final OPType opt = OPType.minimum;
    /**
     *
     */
    private static final long serialVersionUID = -6072156179108651118L;
    public static MinimumOP op = null;

    public static MinimumOP getInstance() {
        if (op == null) {
            op = new MinimumOP();
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
        tape.getTensorOP().minimum(self, other, y);
        if (self.isRequiresGrad()) {
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
            Tensor dy = tape.getTmp();
            tape.getTensorOP().minimum_back(x, y, dy);
            tape.getTensorOP().mulPlus(delta, dy, x.getGrad());
        }
        if (y != null && y.isRequiresGrad()) {
            Tensor dy = tape.getTmp();
            tape.getTensorOP().maximum_back(x, y, dy);
            tape.getTensorOP().mulPlus(delta, dy, y.getGrad());
        }
    }
}

