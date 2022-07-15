package com.omega.engine.autograd.operater;

import java.io.Serializable;

import com.omega.engine.autograd.data.Tensor;
import com.omega.engine.autograd.operater.type.OPType;

public abstract class OP implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -3099365049109802809L;
	
	private OPType opType;
	
	public abstract void backward(Tensor seft);

	public OPType getOpType() {
		return opType;
	}

	public void setOpType(OPType opType) {
		this.opType = opType;
	}
	
}
