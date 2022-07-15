package com.omega.engine.gpu.data;

import java.util.ArrayList;
import java.util.List;

public class CacheDataSet {
	
	public int number = 0;

	public CacheDataSet(int number) {
		this.number = number;
	}
	
	private List<float[]> dim1dSet = new ArrayList<float[]>();
	
	private List<float[][]> dim2dSet = new ArrayList<float[][]>();;
	
	private List<float[][][]> dim3dSet = new ArrayList<float[][][]>();;
	
	private List<float[][][][]> dim4dSet = new ArrayList<float[][][][]>();

	public List<float[]> getDim1dSet() {
		return dim1dSet;
	}

	public List<float[][]> getDim2dSet() {
		return dim2dSet;
	}

	public List<float[][][]> getDim3dSet() {
		return dim3dSet;
	}

	public List<float[][][][]> getDim4dSet() {
		return dim4dSet;
	};
	
}
