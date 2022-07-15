package com.omega.engine.autograd.operater;

import com.omega.engine.autograd.data.Tensor;
import com.omega.engine.autograd.exceptions.AutogradException;
import com.omega.engine.autograd.operater.type.OPType;

public abstract class FunctionOP extends OP {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1210481070644849618L;
	
	public FunctionOP() {
		this.setOpType(OPType.Function);
	}
	
	public abstract Tensor forward(Tensor left, float e) throws AutogradException;

}
