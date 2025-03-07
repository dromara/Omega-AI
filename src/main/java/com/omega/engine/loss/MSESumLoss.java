package com.omega.engine.loss;

import com.omega.common.data.Tensor;
import com.omega.common.data.Tensors;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.loss.gpu.MSESumLossKernel;
import com.omega.engine.nn.network.Network;

/**
 * Square loss
 * @author Administrator
 * @loss: ∑ (y - f(x))^2
 * @diff: 2 * (y - f(x))
 */
public class MSESumLoss extends LossFunction {
	
	public final LossType lossType = LossType.MSE;
	
	private static MSESumLoss instance;
	
	private MSESumLossKernel kernel;
	
	private Tensor loss;
	
	private Tensor diff;
	
	public static MSESumLoss operation(CUDAManager cudaManager) {
		if(instance == null) {
			instance = new MSESumLoss(cudaManager);
		}
		return instance;
	}
	
	public MSESumLoss(Network network) {
		setNet(network);
		kernel = new MSESumLossKernel(network.cudaManager);
	}
	
	public MSESumLoss(CUDAManager cudaManager) {
		kernel = new MSESumLossKernel(cudaManager);
	}
	
	public void init(Tensor input) {
		if(loss == null || loss.number != input.number) {
			this.loss = new Tensor(input.number, 1, 1, 1, true);
//			this.output = new Tensor(input.number, input.channel, input.height, input.width, true);
			this.diff = new Tensor(input.number, input.channel, input.height, input.width, true);
		}
	}
	
//	public static MSELoss operation() {
//		if(instance == null) {
//			instance = new MSELoss();
//		}
//		return instance;
//	}

	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.MSE;
	}
	
	@Override
	public Tensor loss(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		init(x);
//		x.showDM();
//		x.showDMByOffset(0, 100);
//		label.showDMByOffset(0, 100);
		kernel.forward(x, label, loss);
//		loss.showDMByOffset(0, 4);
//		loss.showDM();
//		x.setRequiresGrad(true);
//		Graph.start();
//		Tensor loss = label.sub(x).pow(2.0f).div(2.0f).sum(0).div(x.number);
		return loss;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		kernel.backward(x, label, diff);
		return diff;
//		Graph.clearGrad();
//		Graph.backward();
//		return x.getGrad();
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

	
	public static void main(String[] args) {
		CUDAManager cudaManager = new CUDAManager(0);
		int N = 3;
		int W = 4;
		float[] x = MatrixUtils.order(N * W, 1, 1);
		Tensor xt = Tensors.tensor(N, 1, 1, W, x, true);
		float[] label = MatrixUtils.order(N * W, 0.1f, 0.1f);
		Tensor labelt = Tensors.tensor(N, 1, 1, W, label, true);
		Tensor loss = MSESumLoss.operation(cudaManager).loss(xt, labelt);
		
		loss.showDM();
		Tensor diff = MSESumLoss.operation(cudaManager).diff(xt, labelt);
		diff.showDM();

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
