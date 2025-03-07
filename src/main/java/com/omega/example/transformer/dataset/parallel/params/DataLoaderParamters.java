package com.omega.example.transformer.dataset.parallel.params;

import java.io.Serializable;

import com.omega.common.data.Tensor;

public abstract class DataLoaderParamters implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = -8766234770753552902L;
	
	private Tensor input;

	public Tensor getInput() {
		return input;
	}

	public void setInput(Tensor input) {
		this.input = input;
	}
	
}
