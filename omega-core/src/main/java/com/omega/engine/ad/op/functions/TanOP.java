package com.omega.engine.ad.op.functions;

import com.omega.common.tensor.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;

/**
 * TanOP
 *
 * @author Administrator
 */
public class TanOP extends FunctionOP {
    public static final OPType opt = OPType.tan;
    /**
     *
     */
    private static final long serialVersionUID = -7252060328891832266L;
    public static TanOP op = null;

    public static TanOP getInstance() {
        if (op == null) {
            op = new TanOP();
            op.setOpType(opt);
        }
        return op;
    }

    @Override
    public Tensor forward(Tape tape) {
        // TODO Auto-generated method stub
        Tensor self = tape.getX();
        Tensor y = tape.getOutput();
        tape.getTensorOP().tan(self, y);
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
            tape.getTensorOP().tan_back(x, dy);
            tape.getTensorOP().mulPlus(delta, dy, x.getGrad());
        }
    }
}

