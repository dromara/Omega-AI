package com.omega.engine.autograd.operater.functions;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.autograd.data.Tensor;
import com.omega.engine.autograd.exceptions.AutogradException;
import com.omega.engine.autograd.operater.FunctionOP;

/**
 * power operation
 * @author Administrator
 *
 */
public class PowOP extends FunctionOP {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8805696344353155966L;

	@Override
	public Tensor forward(Tensor left,float e) throws AutogradException {
		// TODO Auto-generated method stub
		return new Tensor(MatrixOperation.pow(left.getData(), e), left, e, this, true);
	}

	@Override
	public void backward(Tensor seft) {
		// TODO Auto-generated method stub
		
		if(seft.getLeftTensor()!=null && seft.getLeftTensor().isRequiresGrad()) {
			/**
			 * leftGrad += e * leftData ^ (e-1) * grad
			 */
			float[][][][] thisTensor = MatrixOperation.multiplication(MatrixOperation.pow(seft.getLeftTensor().getData(), seft.getE() - 1), seft.getE());
			thisTensor =  MatrixOperation.multiplication(thisTensor, seft.getGrad());
			seft.getLeftTensor().setGrad(MatrixOperation.add(thisTensor, seft.getGrad()));
		}
		
	}

}
