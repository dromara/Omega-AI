package com.omega.common.utils;

/**
 * label and output transformer
 * 
 * @author Administrator
 *
 */
public class LabelUtils {
	
	/**
	 * labelToVector
	 * @param label
	 * @param labelSet
	 * @return
	 */
	public static float[] labelIndexToVector(int labelIndex,int labelSize) {
		
		float[] vector = new float[labelSize];
		
		for(int i = 0;i<labelSize;i++) {
			if(i == labelIndex) {
				vector[i] = 1.0f;
			}
		}
		
		return vector;
	}
	
	/**
	 * labelToVector
	 * @param label
	 * @param labelSet
	 * @return
	 */
	public static float[] labelToVector(String label,String[] labelSet) {
		
		float[] vector = new float[labelSet.length];
		
		for(int i = 0;i<labelSet.length;i++) {
			if(labelSet[i].equals(label)) {
				vector[i] = 1.0f;
			}
		}
		
		return vector;
	}
	
	/**
	 * labelToVector
	 * @param label
	 * @param labelSet
	 * @return
	 */
	public static float[][] labelToVector(String[] label,String[] labelSet) {
		
		float[][] vector = new float[label.length][labelSet.length];
		
		for(int i = 0;i<label.length;i++) {
			for(int j = 0;j<labelSet.length;j++) {
				if(labelSet[j].equals(label[i])) {
					vector[i][j] = 1.0f;
				}
			}
		}
		
		return vector;
	}
	
	/**
	 * labelToVector
	 * @param label
	 * @param labelSet
	 * @return
	 */
	public static float[][] labelToVector(float[] label,float[] labelSet) {
		
		float[][] vector = new float[label.length][labelSet.length];
		
		for(int i = 0;i<label.length;i++) {
			for(int j = 0;j<labelSet.length;j++) {
				if(labelSet[j] == label[i]) {
					vector[i][j] = 1.0f;
				}
			}
		}
		
		return vector;
	}
	
	/**
	 * labelToVector
	 * @param label
	 * @return
	 */
	public static float[] labelToVector(String label) {
		
		float[] vector = new float[1];
		
		if(label != null) {
			vector[0] = Float.parseFloat(label);
		}
		
		return vector;
	}
	
	/**
	 * vectorTolabel
	 * @param label
	 * @param labelSet
	 * @return
	 */
	public static String vectorTolabel(float[] vector,String[] labelSet) {
//		System.out.println(JsonUtils.toJson(vector));
		int index = MatrixOperation.maxIndex(vector);
		
		return labelSet[index];
	}
	
	/**
	 * vectorTolabel
	 * @param label
	 * @param labelSet
	 * @return
	 */
	public static String vectorTolabel(float[][][] vector,String[] labelSet) {
//		System.out.println(JsonUtils.toJson(vector));
		int index = MatrixOperation.maxIndex(vector);
		
		return labelSet[index];
	}
	
	/**
	 * checkLabelForVector
	 * @param output
	 * @param labelSet
	 * @param label
	 * @return
	 */
	public static boolean checkLabelForVector(float[] output,String[] labelSet,String label) {
		
		String predictLabel = LabelUtils.vectorTolabel(output, labelSet);
		
		if(!label.equals(predictLabel)) {
			return false;
		}
		
		return true;
	}
	
}
