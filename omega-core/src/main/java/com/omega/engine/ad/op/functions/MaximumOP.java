package com.omega.engine.ad.op.functions;

import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.tensor.Tensor;

/**
 * MaximumOP
 *
 * @author Administrator
 */
public class MaximumOP extends FunctionOP {
    public static final OPType opt = OPType.maximum;
    /**
     *
     */
    private static final long serialVersionUID = -6072156179108651118L;
    public static MaximumOP op = null;

    public static MaximumOP getInstance() {
        if (op == null) {
            op = new MaximumOP();
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
        tape.getTensorOP().maximum(self, other, y);
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
            tape.getTensorOP().maximum_back(x, y, dy);
            tape.getTensorOP().mulPlus(delta, dy, x.getGrad());
        }
        if (y != null && y.isRequiresGrad()) {
            Tensor dy = tape.getTmp();
            tape.getTensorOP().minimum_back(x, y, dy);
            tape.getTensorOP().mulPlus(delta, dy, y.getGrad());
        }
    }
}

