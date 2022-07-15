package com.omega.engine.autograd.operater;

import com.omega.engine.autograd.data.Tensor;
import com.omega.engine.autograd.exceptions.AutogradException;
import com.omega.engine.autograd.operater.type.OPType;

public abstract class SignOP extends OP {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8405087989735183631L;
	
	public SignOP() {
		this.setOpType(OPType.Sgin);
	}
	
	public abstract Tensor forward(Tensor left, Tensor right) throws AutogradException;

}
