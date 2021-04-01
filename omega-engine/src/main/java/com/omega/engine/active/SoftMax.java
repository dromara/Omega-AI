//package com.omega.engine.active;
//
//import com.omega.common.utils.JsonUtils;
//import com.omega.common.utils.MatrixOperation;
//
///**
// * 
// * @ClassName: SoftMax
// *
// * @author lijiaming
// *
// * @date 2020年7月27日
// *
// * @Description: 
// * TODO(用一句话描述该文件做什么)
// * SoftMax
// * active: e ^ x[i] / ∑ e ^ x
// *         防止数值溢出，需每项减出最大值
// *         e ^ (x[i] - max) / ∑ e ^ (x - max)
// *         
// * diff: i = j : si - sj^2
// *       i != j : -si * sj        
// */
//public class Softmax extends ActiveFunction {
//	
//	public Softmax(){
//		this.activeType = ActiveType.softmax;
//	}
//	
//	@Override
//	public double[] active(double[] x) {
//		// TODO Auto-generated method stub
//		this.input = MatrixOperation.clone(x);
//		this.output = MatrixOperation.zero(x.length);
//		double max = MatrixOperation.max(this.input);
//		double[] temp = MatrixOperation.subtraction(this.input, max);
//		temp = MatrixOperation.exp(temp);
//		double sum = MatrixOperation.sum(temp);
//		for(int i = 0;i<temp.length;i++) {
//			this.output[i] = temp[i] / sum;
//		}
//		return this.output;
//	}
//	
//	/**
//	 * i = j : si - sj^2
//     * i != j : - si * sj 
//	 */
//	@Override
//	public double[] diff() {
//		// TODO Auto-generated method stub
//		this.diff = MatrixOperation.clone(this.active(this.input));
//		
//		double[][] temp = MatrixOperation.zero(this.diff.length, this.diff.length);
//		
//		for(int i = 0;i<this.diff.length;i++) {
//			for(int j = 0;j<this.diff.length;j++) {
//				if(i == j) {
//					temp[i][j] = (1d - this.output[i]) * this.output[i];
//				}else{
//					temp[i][j] = -1d * this.output[i] * this.output[j];
//				}
//				
//			}
//		}
//		
//		System.out.println(JsonUtils.toJson(temp));
//		
//		return null;
//	}
//	
//	@Override
//	public double[] activeTemp(double[] x) {
//		// TODO Auto-generated method stub
//		double[] output = MatrixOperation.zero(x.length);
//		double max = MatrixOperation.max(x);
//		double[] temp = MatrixOperation.subtraction(x, max);
//		temp = MatrixOperation.exp(temp);
//		double sum = MatrixOperation.sum(temp);
//		for(int i = 0;i<temp.length;i++) {
//			output[i] = temp[i] / sum;
//		}
//		return output;
//	}
//
//	/**
//	 * i = j : si - sj^2
//     * i != j : -si * sj 
//	 */
//	@Override
//	public double[] diffTemp(double[] x) {
//		// TODO Auto-generated method stub
//		double[] diff = this.activeTemp(x);
//		double[][] temp = MatrixOperation.zero(diff.length, diff.length);
//		
//		for(int i = 0;i<diff.length;i++) {
//			for(int j = 0;j<diff.length;j++) {
//				if(i == j) {
//					temp[i][j] = (1d - diff[i]) * diff[i];
//				}else{
//					temp[i][j] = -1d * diff[i] * diff[j];
//				}
//				
//			}
//		}
//		
//		System.out.println(JsonUtils.toJson(temp));
//		return null;
//	}
//	
//	public static void main(String[] args) {
//		Softmax function = new Softmax();
//		double[] x = new double[] {0.1,-0.03,0.025,-0.4,0.807,-0.12,0.001,0.002,0.2,0.5};
//		System.out.println(JsonUtils.toJson(function.activeTemp(x)));
//		function.diffTemp(x);
////		System.out.println(JsonUtils.toJson(function.diffTemp(x)));
//		
////		
//		double error = function.gradientCheck(x);
////		System.out.println("error:"+error);
//	}
//
//}
