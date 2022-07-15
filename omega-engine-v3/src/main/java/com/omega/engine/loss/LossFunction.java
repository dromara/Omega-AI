package com.omega.engine.loss;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;

public abstract class LossFunction {
	
	public LossType lossType;
	
	public float eta = 0.00001f;
	
	public abstract Blob loss(Blob x, float[][] label);
	
	public abstract Blob diff(Blob x, float[][] label);
	
	public abstract LossType getLossType();
	
	public float gradientCheck(Blob x, float[][] label) {
		Blob diff = this.diff(x,label);
		Blob f1 = this.loss(Blobs.blob(MatrixOperation.add(x.maxtir, eta)), label);
		Blob f2 = this.loss(Blobs.blob(MatrixOperation.subtraction(x.maxtir, eta)), label);
		float[][][][] temp = MatrixOperation.subtraction(f1.maxtir, f2.maxtir);
		temp = MatrixOperation.division(temp, 2 * eta);
		System.out.println("diff:"+JsonUtils.toJson(diff));
		System.out.println("gradientCheck:"+JsonUtils.toJson(temp));
		float[][][][] error = MatrixOperation.subtraction(diff.maxtir, temp);
		return MatrixOperation.sum(error);
	} 
	
}
