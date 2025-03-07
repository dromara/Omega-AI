package com.omega.engine.ad.op.sign;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.SignOP;

/**
 * f(a,b) = a * b;
 * da,db = g*b,a*g
 * @author Administrator
 *
 */
public class MulOP extends SignOP{

	/**
	 * 
	 */
	private static final long serialVersionUID = -4667315516225564503L;
	
	public static MulOP op = null;
	
	public static final OPType opt = OPType.multiplication;
	
	public static MulOP getInstance() {
		if(op == null) {
			op = new MulOP();
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
		if(other != null) {
			tape.getTensorOP().mul(self, other, y);
		}else {
			tape.getTensorOP().mul(self, tape.getScalar(), y);
		}
		if(self.isRequiresGrad() || (other != null && other.isRequiresGrad())) {
			y.setRequiresGrad(true);
		}
		return y;
	}

	@Override
	public void backward(Tensor delta, Tape tape) {
		// TODO Auto-generated method stub
		Tensor x = tape.getX();
		Tensor y = tape.getY();
		if(x.isRequiresGrad()) {
			if(y != null) {
				tape.getTensorOP().mulPlus(delta, y, x.getGrad());
			}else {
				tape.getTensorOP().mulPlus(delta, tape.getScalar(), x.getGrad());
			}
		}
		if(y !=null && y.isRequiresGrad()) {
			tape.getTensorOP().mulPlus(delta, x, y.getGrad());
		}
	}

}
