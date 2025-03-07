package com.omega.engine.ad.op.data;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OP;
import com.omega.engine.ad.op.OPType;

/**
 * 获取指定向量数据
 * @author Administrator
 *
 */
public class SetOP extends OP{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 7010180428917414516L;

	public static SetOP op = null;
	
	public static final OPType opt = OPType.set;
	
	public static SetOP getInstance() {
		if(op == null) {
			op = new SetOP();
			op.setOpType(opt);
		}
		return op;
	}
	
	@Override
	public Tensor forward(Tape tape) {
		Tensor self = tape.getX();
		Tensor y = tape.getY();
		setByPosition(self, y, tape.getPosition(), tape);
		if(self.isRequiresGrad()) {
			y.setRequiresGrad(true);
		}
		return y;
	}
	
	@Override
	public void backward(Tensor delta, Tape tape) {
		// TODO Auto-generated method stub
		Tensor y = tape.getY();
		if(y.isRequiresGrad()) {
			addByPosition(y.getGrad(), delta, tape.getPosition(), tape);
		}
	}
	
	public void addByPosition(Tensor a,Tensor b,int[] position, Tape tape) {
		int dims = position[0];
		int start = position[1];
		if(a.isHasGPU()) {
			switch (dims) {
			case 0:
				tape.getTensorOP().op.axpy_gpu(b, a, start * a.channel * a.height * a.width, 0);
				break;
			}
		}else {
			int n = a.getNumber();
			int c = a.getChannel();
			int h = a.getHeight();
			int w = a.getWidth();
			MatrixOperation.add(b.data, a.data, n, c, h, w, position);
		}
	}
	
	public void setByPosition(Tensor org,Tensor target,int[] position, Tape tape) {
		
		int dims = position[0];
		int start = position[1];
		
		switch (dims) {
		case 0:
			setByNumber(org, target, start, tape);
			break;
		case 1:

			break;
		default:
			break;
		}
		
	}

	public void setByNumber(Tensor org,Tensor target,int start, Tape tape) {
		
		assert org.getNumber() >= (start - 1);
		
		if(org.isHasGPU()) {
			tape.getTensorOP().op.copy_gpu(target, org, 0, start * target.channel * target.height * target.width);
		}else {
			System.arraycopy(target.data, 0, org.data, start * target.channel * target.height * target.width, target.dataLength);
		}
	}
	
}
