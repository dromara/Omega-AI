package com.omega.engine.ad.op.sign;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.SignOP;


/**
 * f(scalar,b) = scalar / b;
 * db = -g * scalar / b^2
 * @author Administrator
 */
public class ScalarDivOP extends SignOP {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3087002822041265440L;

	public static ScalarDivOP op = null;
	
	public static final OPType opt = OPType.scalarDivision;
	
	public static ScalarDivOP getInstance() {
		if(op == null) {
			op = new ScalarDivOP();
			op.setOpType(opt);
		}
		return op;
	}
	
	@Override
	public Tensor forward(Tape tape) {
		// TODO Auto-generated method stub
		Tensor self = tape.getX();
		Tensor y = tape.getOutput();
		tape.getTensorOP().div(tape.getScalar(), self, y);
		if(self.isRequiresGrad()) {
			y.setRequiresGrad(true);
		}
		return y;
	}

	@Override
	public void backward(Tensor delta, Tape tape) {
		// TODO Auto-generated method stub
		Tensor x = tape.getX();
		if(x.isRequiresGrad()) {
			if(x.getGrad().isHasGPU()) {
				tape.getTensorOP().op.div_scalar_bGrad_gpu(delta, tape.getScalar(), x, x.getGrad());
			}else {
				bGrad(delta.data, tape.getScalar(), x.data, x.getGrad().data);
			}
		}
	}
	
	/**
	 * db = -delta * a / b^2
	 * @param delta
	 * @param a
	 * @param b
	 * @return
	 */
	public static void bGrad(float[] delta,float a,float[] b,float[] grad) {
		for(int i = 0;i<delta.length;i++){
			grad[i] += - 1.0f * delta[i] * a / (b[i] * b[i]); 
		}
	}
	
}
