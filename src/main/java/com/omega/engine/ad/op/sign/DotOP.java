package com.omega.engine.ad.op.sign;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.SignOP;

public class DotOP extends SignOP {


	/**
	 * 
	 */
	private static final long serialVersionUID = 8033645023767349745L;
	
	public static DotOP op = null;
	
	public static final OPType opt = OPType.dot;
	
	public static DotOP getInstance() {
		if(op == null) {
			op = new DotOP();
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
		tape.getTensorOP().dot(self, other, y);
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
			tape.getTensorOP().dotDX(delta, y, x.getGrad());
		}
		if(y !=null && y.isRequiresGrad()) {
			tape.getTensorOP().dotDW(x, delta, y.getGrad());
		}
	}

}
