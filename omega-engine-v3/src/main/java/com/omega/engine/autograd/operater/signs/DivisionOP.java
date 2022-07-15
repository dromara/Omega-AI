package com.omega.engine.autograd.operater.signs;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.autograd.data.Tensor;
import com.omega.engine.autograd.exceptions.AutogradException;
import com.omega.engine.autograd.operater.SignOP;

/**
 * Add operation
 * @author Administrator
 *
 */
public class DivisionOP extends SignOP {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8805696344353155966L;

	@Override
	public Tensor forward(Tensor left,Tensor right) throws AutogradException {
		// TODO Auto-generated method stub
		return new Tensor(MatrixOperation.division(left.getData(), right.getData()), left, right, this, true);
	}

	@Override
	public void backward(Tensor seft) {
		// TODO Auto-generated method stub
		
		if(seft.getLeftTensor()!=null && seft.getLeftTensor().isRequiresGrad()) {
			/**
			 * leftGrad += 1 / rightData * grad
			 */
			float[][][][] thisTensor = MatrixOperation.division(seft.getGrad(), seft.getRightTensor().getData());
			seft.getLeftTensor().setGrad(MatrixOperation.add(seft.getLeftTensor().getGrad(), thisTensor));
		}
		
		if(seft.getRightTensor()!=null && seft.getRightTensor().isRequiresGrad()) {
			/**
			 * rightGrad += -1 * leftData / rigthData^2 * grad
			 */
			float[][][][] thisTensor = MatrixOperation.multiplication(seft.getRightTensor().getData(), seft.getRightTensor().getData());
			thisTensor = MatrixOperation.division(seft.getLeftTensor().getData(), thisTensor);
			thisTensor =  MatrixOperation.multiplication(thisTensor, seft.getGrad());
			seft.getRightTensor().setGrad(MatrixOperation.subtraction(seft.getRightTensor().getGrad(), thisTensor));
		}
		
	}

}
