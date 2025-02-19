package com.omega.engine.loss;

import com.omega.common.data.Tensor;
import com.omega.common.data.Tensors;
import com.omega.engine.ad.Graph;
import com.omega.engine.loss.gpu.BCELossKernel;

/**
 * 二分类loss
 * @author Administrator
 *
 */
public class BCELoss extends LossFunction {

	public final LossType lossType = LossType.BCE;
	
	private static BCELoss instance;
	
	private BCELossKernel kernel;
	
	private Tensor loss;
	
	private Tensor diff;
	
	private Graph g;
	
	public static BCELoss operation() {
		if(instance == null) {
			instance = new BCELoss();
		}
		return instance;
	}
	
	public BCELoss() {
		kernel = new BCELossKernel();
		g = new Graph();
	}
	
	public void init(Tensor input) {
		if(loss == null || loss.number != input.number) {
			this.loss = new Tensor(input.number, 1, 1, 1, true);
			this.diff = new Tensor(input.number, input.channel, input.height, input.width, true);
		}
	}
	
	public void initGraph(Tensor x,Tensor label) {
		if(x.getG() == null){
			x.setG(g);
		}
		if(label.getG() == null){
			label.setG(g);
		}
	}

//	@Override
//	public Tensor loss(Tensor x, Tensor label) {
//		// TODO Auto-generated method stub
////		x.showDM();
//		initGraph(x, label);
//		x.setRequiresGrad(true);
////		x.getG().start();
//		Tensor loss1 = label.scalarSub(1.0f).mul(x.scalarSub(1.0f).log());
//		Tensor loss = loss1.add(label.mul(x.log()));
//		return loss.sum(0).div(-x.number * x.width);
//	}
	
	@Override
	public Tensor loss(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		init(x);
		kernel.forward(x, label, loss);
		return loss;
	}

//	@Override
//	public Tensor diff(Tensor x, Tensor label) {
//		// TODO Auto-generated method stub
//		x.getG().clearGrad();
//		x.getG().backward();
//		return x.getGrad();
//	}

	@Override
	public Tensor diff(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		kernel.backward(x, label, diff);
		return diff;
	}
	
	@Override
	public Tensor[] loss(Tensor[] x, Tensor label) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor[] diff(Tensor[] x, Tensor label) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.BCE;
	}
	
	public static void main(String[] args) {
		float[] x = new float[] {0.99952507f,0.9999833f,1.0f,1.0f,1.0f,1.27869424E-20f,1.0f,3.8254528E-26f};
		Tensor xt = Tensors.tensor(8, 1, 1, 1, x, true);
		float[] label = new float[] {1,1,1,1,1,1,1,1,1};
		Tensor labelt = Tensors.tensor(8, 1, 1, 1, label, true);
//		Tensor a = sigmoid(xt);
//		a.showDM();
		Tensor loss = BCELoss.operation().loss(xt, labelt);
		loss.showDM();
		Tensor diff = BCELoss.operation().diff(xt, labelt);
		diff.showDM();
//		Graph.clearGrad();
//		Graph.backward();
//		xt.getGrad().showDM();
//		float error = BCELoss.operation().gradientCheck(xt,labelt);
//		System.out.println("error:"+error);
	}

	@Override
	public Tensor loss(Tensor x, Tensor label, Tensor loss) {
		// TODO Auto-generated method stub
		init(x);
		kernel.forward(x, label, loss);
		return loss;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label, Tensor diff) {
		// TODO Auto-generated method stub
		kernel.backward(x, label, diff);
		return diff;
	}

	@Override
	public Tensor loss(Tensor x, Tensor label, int igonre) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label, int igonre) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label, int igonre, int count) {
		// TODO Auto-generated method stub
		return null;
	}
	
}
