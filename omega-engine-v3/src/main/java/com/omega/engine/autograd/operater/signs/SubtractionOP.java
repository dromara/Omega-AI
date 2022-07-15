package com.omega.engine.autograd.operater.signs;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.autograd.data.Tensor;
import com.omega.engine.autograd.exceptions.AutogradException;
import com.omega.engine.autograd.operater.SignOP;

/**
 * Subtraction operation
 * @author Administrator
 *
 */
public class SubtractionOP extends SignOP {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8805696344353155966L;

	@Override
	public Tensor forward(Tensor left,Tensor right) throws AutogradException {
		// TODO Auto-generated method stub
		return new Tensor(MatrixOperation.subtraction(left.getData(), right.getData()), left, right, this, true);
	}

	@Override
	public void backward(Tensor seft) {
		// TODO Auto-generated method stub
		
		if(seft.getLeftTensor()!=null && seft.getLeftTensor().isRequiresGrad()) {
			seft.getLeftTensor().setGrad(MatrixOperation.add(seft.getLeftTensor().getGrad(), seft.getGrad()));
		}
		
		if(seft.getRightTensor()!=null && seft.getRightTensor().isRequiresGrad()) {
			seft.getRightTensor().setGrad(MatrixOperation.subtraction(seft.getRightTensor().getGrad(), seft.getGrad()));
		}
		
	}

}
