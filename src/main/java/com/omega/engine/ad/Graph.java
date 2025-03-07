package com.omega.engine.ad;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.OP;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.ad.op.data.GetOP;
import com.omega.engine.ad.op.data.SetOP;
import com.omega.engine.ad.op.functions.ATanOP;
import com.omega.engine.ad.op.functions.ClampOP;
import com.omega.engine.ad.op.functions.CosOP;
import com.omega.engine.ad.op.functions.ExpOP;
import com.omega.engine.ad.op.functions.LogOP;
import com.omega.engine.ad.op.functions.MaxOP;
import com.omega.engine.ad.op.functions.MaximumOP;
import com.omega.engine.ad.op.functions.MinimumOP;
import com.omega.engine.ad.op.functions.PowOP;
import com.omega.engine.ad.op.functions.SinOP;
import com.omega.engine.ad.op.functions.SumOP;
import com.omega.engine.ad.op.functions.TanOP;
import com.omega.engine.ad.op.functions.TransposeOP;
import com.omega.engine.ad.op.sign.AddOP;
import com.omega.engine.ad.op.sign.DivOP;
import com.omega.engine.ad.op.sign.DotOP;
import com.omega.engine.ad.op.sign.MulOP;
import com.omega.engine.ad.op.sign.ScalarDivOP;
import com.omega.engine.ad.op.sign.ScalarSubOP;
import com.omega.engine.ad.op.sign.SubOP;
import com.omega.engine.gpu.CUDAManager;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;

/**
 * 计算图
 * @author Administrator
 *
 */
public class Graph{
	
	/**
	 * 计算图map
	 */
	private List<Tape> tapes = new ArrayList<Tape>();
	
	public int tapeIndex = 0;
	
	private boolean lock = false;
	
	private TensorOP tensorOP;
	
	public Graph(TensorOP tensorOP) {
		this.tensorOP = tensorOP;
	}
	
	public void start() {
		tapeIndex = 0;
	}
	
	public void lock() {
		lock = true;
	}
	
	public void unlock() {
		lock = false;
	}
	
	public void showGraph() {
		for(int i = 0;i<tapes.size();i++) {
			System.out.println(i+":["+tapes.get(i).getOp().getOpType()+"]");
			System.out.println("x1:["+tapes.get(i).getX()+"]|x2:["+tapes.get(i).getY()+"]|out:["+tapes.get(i).getOutput()+"]");
		}
	}
	
	public void reset() {
		this.tapes.clear();
	}
	
	public void clearGrad() {
		for(int i = 0;i<tapes.size();i++) {
			this.tapes.get(i).zeroGrad();
		}
//		reset();
	}
	
	public void add(Tape tape) {
		this.tapes.add(tape);
	}
	
	public Tape getTape(OP op,Tensor self,Tensor other,float scalar,float constant,int[] position) {
		Tape tape = null;
		if(!lock) {
			tape = new Tape(op, self, other, scalar, constant, position, this);
//			System.out.println(tape.getOp().getOpType()+":"+tape.isSub());
			checkSubTape(self, other);
			
			this.add(tape);
		}else {
			tape = tapes.get(tapeIndex);
			if(tape.getOp().getOpType().equals(OPType.sum)) {
				tape.getOutput().fill(0.0f, tape.getTensorOP().op);
			}
			tapeIndex++;
		}
		return tape;
	}
	
	public void checkSubTape(Tensor a,Tensor b) {
		for(Tape tape:tapes) {
			if(!tape.isSub() && (tape.getX() == a || tape.getY() == a || tape.getX() == b || tape.getY() == b ||
					tape.getOutput() == a || tape.getOutput() == b)) {
//				System.out.println(this.toString()+":"+tape.getOp().getOpType().toString());
				tape.setSub(true);
			}
		}
	}
	
	public Tensor OP(OPType opType,Tensor self,Tensor other) {
		OP op = null;
		switch (opType) {
		case add:
			op = AddOP.getInstance();
			break;
		case subtraction:
			op = SubOP.getInstance();
			break;
		case multiplication:
			op = MulOP.getInstance();
			break;
		case division:
			op = DivOP.getInstance();
			break;
		case maximum:
			op = MaximumOP.getInstance();
			break;
		case minimum:
			op = MinimumOP.getInstance();
			break;
		case dot:
			op = DotOP.getInstance();
			break;
		default:
			break;	
		}
		
		if(op == null) {
			throw new RuntimeException("the op is not support.");
		}
		
		Tape tape = this.getTape(op, self, other, 0, 0, null);
		Tensor output = tape.forward();
		output.setG(this);
		return output;
	}
	
	public Tensor OP(OPType opType,Tensor self,float other) {
		
		OP op = null;
		
		switch (opType) {
		case add:
			op = AddOP.getInstance();
			break;
		case subtraction:
			op = SubOP.getInstance();
			break;
		case scalarSubtraction:
			op = ScalarSubOP.getInstance();
			break;
		case multiplication:
			op = MulOP.getInstance();
			break;
		case division:
			op = DivOP.getInstance();
			break;
		case scalarDivision:
			op = ScalarDivOP.getInstance();
			break;
		case pow:
			op = PowOP.getInstance();
			break;
		default:
			break;	
		}
		
		if(op == null) {
			throw new RuntimeException("the op is not support.");
		}
		
		Tape tape = getTape(op, self, null, other, 0, null);
		Tensor output = tape.forward();
		output.setG(this);
		return output;
	}
	
	public Tensor OP(OPType opType,Tensor self,float constant1,float constant2) {
		
		OP op = null;
		
		switch (opType) {
		case clamp:
			op = ClampOP.getInstance();
			break;
		default:
			break;	
		}
		
		if(op == null) {
			throw new RuntimeException("the op is not support.");
		}
		
		Tape tape = getTape(op, self, null, constant1, constant2, null);
		Tensor output = tape.forward();
		output.setG(this);
		return output;
	}
	
	public Tensor OP(OPType opType,Tensor self) {
		
		OP op = null;
		
		switch (opType) {
		case log:
			op = LogOP.getInstance();
			break;
		case sin:
			op = SinOP.getInstance();
			break;
		case cos:
			op = CosOP.getInstance();
			break;
		case tan:
			op = TanOP.getInstance();
			break;
		case atan:
			op = ATanOP.getInstance();
			break;
		case exp:
			op = ExpOP.getInstance();
			break;
		case transpose:
			op = TransposeOP.getInstance();
			break;
		default:
			break;	
		}
		
		if(op == null) {
			throw new RuntimeException("the op is not support.");
		}
		
		Tape tape = this.getTape(op, self, null, 0, 0, null);
		Tensor output = tape.forward();
		output.setG(this);
		return output;
	}
	
	public Tensor OP(OPType opType,Tensor self,int[] position) {
		OP op = null;
		
		switch (opType) {
		case get:
			op = GetOP.getInstance();
			break;
		case sum:
			op = SumOP.getInstance();
			break;
		case max:
			op = MaxOP.getInstance();
			break;
		default:
			break;	
		}
		
		if(op == null) {
			throw new RuntimeException("the op is not support.");
		}
		
		Tape tape = getTape(op, self, null, 0, 0, position);
		Tensor output = tape.forward();
		output.setG(this);
		return output;
	}
	
	public Tensor OP(OPType opType,Tensor self,Tensor other,int[] position) {
		OP op = null;
		
		switch (opType) {
		case set:
			op = SetOP.getInstance();
			break;
		default:
			break;	
		}
		
		if(op == null) {
			throw new RuntimeException("the op is not support.");
		}
		
		Tape tape = getTape(op, self, other, 0, 0, position);
		Tensor output = tape.forward();
		output.setG(this);
		return output;
	}
	
	public void backward(Tensor delta) {
//		float[] preDelta = null;
		this.lock = true;
		for(int i = tapes.size() - 1;i >= 0;i--) {
			Tape tape = tapes.get(i);
			if(i == tapes.size() - 1) {
				tape.backward(delta);
			}else {
				tape.backward();
			}
//			preDelta = tape.getInputs().get(0).getGrad();
		}
		this.tapeIndex = 0;
	}
	
	public void backward() {
		this.lock = true;
		for(int i = tapes.size() - 1;i >= 0;i--) {
			Tape tape = tapes.get(i);
			/**
			 * 初始化最后一代的grad
			 */
//			System.out.println(tape.getOp().getOpType().toString()+":"+tape.isSub());
			if(!tape.isSub()) {
				tape.getOutput().getGrad().fill(1.0f, tape.getTensorOP().op);
			}
			tape.backward();
		}
		this.tapeIndex = 0;
	}
	
	public void formula1(){
		int number = 1;
		int channel  = 1;
		int height = 1;
		int width = 1;
		int length = number * channel * height * width;
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.val(length, 2.0f));
		
		Tensor y = new Tensor(number, channel, height, width, MatrixUtils.val(length, 5.0f));
		
		x.setRequiresGrad(true);
		y.setRequiresGrad(true);
		
		for(int i = 0;i<10;i++) {
			
			this.clearGrad();
			
			/**
			 * f(x,y)=ln(x)+x*y−sin(y)
			 */
			Tensor v5 = x.log().add(x.mul(y)).sub(y.sin());
			
			this.backward();
			
//			System.out.println(JsonUtils.toJson(Graph.tapes));
			
			System.out.println("z:"+JsonUtils.toJson(v5.data));
			
			System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
			
			System.out.println("dy:"+JsonUtils.toJson(y.getGrad()));

		}
	}
	
	public void sigmoid_gpu(Tensor x,Tensor y) {
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		y.hostToDevice();
		
		long start = System.nanoTime();
		
		Tensor v1 = x.get(1, 0, 2).mul(-1).exp().add(1).scalarDiv(1);
		
		Tensor v2 = x.get(1, 2, 2);
		
		Tensor v3 = y.get(1, 4, 2).sub(x.get(1, 4, 2).mul(-1).exp().add(1).scalarDiv(1)).pow(2);
		
		Tensor z = v1.add(v2).add(v3);
		
//		Graph.showGraph();
		
		this.backward();

		z.syncHost();
		
//		System.out.println("z:"+JsonUtils.toJson(z.data));
		x.getGrad().syncHost();
//		System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
		
		System.out.println(((System.nanoTime() - start) / 1e6) + "ms.");
		
//		PrintUtils.printImage(x.getGrad());
		
	}
	
	public static void get_gpu() {
		
		int number = 64;
		int channel  = 128;
		int height = 32;
		int width = 32;
		int length = number * channel * height * width;
		
		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.order(length, 0, 1), true);
		
		long start = System.nanoTime();
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
//		x.showDM();
		
		Tensor v1 = x.get(1, 1, 10).pow(2.0f);
		
		Tensor v2 = x.get(1, 14, 10);
		
		graph.showGraph();
		
		graph.backward();
		
		v1.syncHost();
		
		v2.syncHost();
		
		x.getGrad().syncHost();
		
		System.out.println(((System.nanoTime() - start) / 1e6) + "ms.");
		
//		System.out.println("z1:"+JsonUtils.toJson(v1.data));
		
//		PrintUtils.printImage(v1);
//		
//		System.out.println("*********************************************");
//		
//		PrintUtils.printImage(v2);
//
//		System.out.println("++++++++++++++++++++++++++++++++++++++++++++");
//		
//		PrintUtils.printImage(x.getGrad());

	}
	
	public static void pow_gpu() {
		
		int number = 2;
		int channel  = 3;
		int height = 5;
		int width = 5;
		int length = number * channel * height * width;
		
		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.order(length, 0, 1), true);
		
		long start = System.nanoTime();
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		Tensor v1 = x.pow(3);
		
		graph.showGraph();
		
		graph.backward();
		
		v1.syncHost();
		
		System.out.println(((System.nanoTime() - start) / 1e6) + "ms.");

	}
	
	public static void show() {
		
		int n = 10;
		int c  = 5;
		int h = 5;
		int w = 5;
		int length = n * c * h * w;
		int count = 2;
		int start = 1;
		
		Tensor x = new Tensor(n, c, h, w, MatrixUtils.order(length, 0, 1));
		
		Tensor y = new Tensor(x.number, count, x.height, x.width, x.isHasGPU());
		
		for(int i = 0;i<y.dataLength;i++) {
			int bc = y.dataLength / n / h / w;
			int size = bc * h * w;
	    	int tn = i / size;
			int tc = (i / h / w) % bc + start;
			int th = (i / w) % h;
			int tw = i % h;
			int index = tn * c * h * w + tc * h * w + th * w + tw;
	    	y.data[i] = x.data[index];
		}
		
		PrintUtils.printImage(y);
	}
	
	public static void yolov3_loss() {
		
		int number = 3;
		int channel  = 18;
		int height = 5;
		int width = 5;
		int length = number * channel * height * width;
		
//		int classNum = 1;
//		
//		int bboxNum = 3;
		
		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.val(length, 0.6f), true);
		
		Tensor y = new Tensor(number, channel, height, width, MatrixUtils.val(length, 1f), true);
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		y.hostToDevice();
		
		long start = System.nanoTime();
		
		Tensor xy1 = BCELoss(sigmoid(x.get(1, 0, 2)), y.get(1, 0, 2));
		
		Tensor wh1 = MSELoss(x.get(1, 2, 2), y.get(1, 2, 2));
		
		Tensor cc1 = BCELoss(sigmoid(x.get(1, 4, 2)), y.get(1, 4, 2));
		
		Tensor xy2 = BCELoss(sigmoid(x.get(1, 6, 2)), y.get(1, 6, 2));
		
		Tensor wh2 = MSELoss(x.get(1, 8, 2), y.get(1, 8, 2));
		 
		Tensor cc2 = BCELoss(sigmoid(x.get(1, 10, 2)), y.get(1, 10, 2));
		
		Tensor z = xy1.add(wh1).add(cc1).add(xy2).add(wh2).add(cc2);
		
//		Graph.showGraph();
		
		graph.backward();

		z.syncHost();
		
		System.out.println("z:"+JsonUtils.toJson(z.data));
		x.getGrad().syncHost();
		System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
		
		System.out.println(((System.nanoTime() - start) / 1e6) + "ms.");
	
		PrintUtils.printImage(z);
		
		PrintUtils.printImage(x.getGrad());
		
	}
	
	public static Tensor sigmoid(Tensor x) {
		return x.mul(-1).exp().add(1).scalarDiv(1);
	}
	
	public static Tensor tanh(Tensor x) {
		Tensor e = x.mul(-2).exp();
		Tensor t1 = e.scalarSub(1);
		Tensor t2 = e.add(1);
		return t1.div(t2);
	}
	
	public static Tensor MSELoss(Tensor pred,Tensor target) {
		// y = (pred - sub)^2
		return pred.sub(target).pow(2);
	}
	
	public static Tensor BCELoss(Tensor pred,Tensor target) {
		// y = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
		return target.mul(-1).mul(pred.log()).sub(target.scalarSub(1.0f).mul(pred.scalarSub(1.0f).log()));
	}
	
	public static void multiLabelSoftMarginLoss() {
	
		int number = 2;
		int channel  = 1;
		int height = 1;
		int width = 4;
//		int length = number * channel * height * width;
		int C = channel * height * width;

		float[] xa = new float[] {0.2f,0.5f,0,0,0.1f,0.5f,0,0.8f};
		
		float[] ya = new float[] {1,1,0,0,0,1,0,1};
		
		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		Tensor x = new Tensor(number, channel, height, width, xa, true);
		
		Tensor y = new Tensor(number, channel, height, width, ya, true);
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		y.hostToDevice();
		
		/**
		 * -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))
		 */
		for(int i = 0;i<20;i++) {

			long start = System.nanoTime();
			
			Tensor x0 = sigmoid(x).log();
			
			Tensor x1 = sigmoid(x.mul(-1.0f)).log().mul(y.scalarSub(1.0f));
			
			Tensor loss = y.mul(x0).add(x1).mul(-1.0f);
			
			loss = loss.sum(1).div(C).sum(0).div(x.number);
			
			graph.clearGrad();
			
			graph.backward();
			
			loss.syncHost();
			
			System.out.println("loss:"+JsonUtils.toJson(loss.data));
			
			x.getGrad().syncHost();
			System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
			
			System.out.println(((System.nanoTime() - start) / 1e6) + "ms.");
		
			PrintUtils.printImage(x.getGrad());
			
		}
		
	}
	
	public static void multiLabelSoftMarginLoss2() {
		
		int number = 64;
		int channel  = 128;
		int height = 32;
		int width = 32;
		int length = number * channel * height * width;
		int C = channel * height * width;

		float[] xa = RandomUtils.gaussianRandom(length, 0.1f);
		
		float[] ya = RandomUtils.gaussianRandom(length, 0.1f);
		
		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		Tensor x = new Tensor(number, channel, height, width, xa, true, graph);
		
		Tensor y = new Tensor(number, channel, height, width, ya, true, graph);
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		y.hostToDevice();
		
		/**
		 * -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))
		 */
		for(int i = 0;i<200;i++) {

			long start = System.nanoTime();
			
			graph.start();
			
			Tensor x0 = sigmoid(x).log();
			
			Tensor x1 = sigmoid(x.mul(-1.0f)).log().mul(y.scalarSub(1.0f));
			
			Tensor loss = y.mul(x0).add(x1).mul(-1.0f);
			
			loss = loss.sum(1).div(C).sum(0).div(x.number);
			
			graph.lock = true;
			
			graph.clearGrad();
			
			graph.backward();
			
			loss.syncHost();
			
//			System.out.println("loss:"+JsonUtils.toJson(loss.data));
			
//			x.getGrad().syncHost();
//			System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
//			x.getGrad().showDM();
			System.out.println(((System.nanoTime() - start) / 1e6) + "ms.");
		
//			PrintUtils.printImage(x.getGrad());
			
		}
		
	}
	
	public static void sq() {
		
		int number = 3;
		int channel  = 18;
		int height = 5;
		int width = 5;
		int length = number * channel * height * width;
//		int C = channel * height * width;
		
		float[] cpx = RandomUtils.gaussianRandom(length, 0.1f);
		
		float[] cpy = RandomUtils.gaussianRandom(length, 0.1f);
		
		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		Tensor x = new Tensor(number, channel, height, width, cpx, true);
		
		Tensor y = new Tensor(number, channel, height, width, cpy, true);
		
		for(int i = 0;i<20;i++) {

			x.data = RandomUtils.gaussianRandom(length, 0.1f);
			
			y.data = RandomUtils.gaussianRandom(length, 0.1f);
			
			sq_back_cpu(x, y);
			
			x.setRequiresGrad(true);
			
			x.hostToDevice();
			y.hostToDevice();
			
//			Tensor loss1 = y.sub(x).pow(2.0f).div(2.0f);

			graph.clearGrad();
			
			graph.backward();
			
			x.getGrad().syncHost();
			System.out.println("dx_gpu:"+JsonUtils.toJson(x.getGrad().data));
		}

	}
	
	public static void sq_back_cpu(Tensor x,Tensor y) {
		
		Tensor temp = new Tensor(x.number, x.channel, x.height, x.width, true);
		
		for(int i = 0;i<x.getDataLength();i++) {
			temp.data[i] = x.data[i] - y.data[i];
		}
		System.out.println("dx_cpu:"+JsonUtils.toJson(temp.data));
	}
	
	public static void sum() {
		
		int number = 3;
		int channel  = 18;
		int height = 5;
		int width = 5;
		int length = number * channel * height * width;

		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.val(length, 0.6f), true, graph);
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		Tensor z = x.sum(1);
		
		graph.backward();
		
		z.syncHost();
		
		System.out.println("z:"+JsonUtils.toJson(z.data));
		
		x.getGrad().syncHost();
		System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
		
		PrintUtils.printImage(x.getGrad());
		
	}
	
	public static void maximum() {
		
		int number = 1;
		int channel  = 1;
		int height = 1;
		int width = 5;

		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		float[] t1 = new float[] {0.1f,1,0.06f,-1,1.3f};
		
		float[] t2 = new float[] {-0.1f,1,0.07f,-1.2f,0.003f};
		
		Tensor b1 = new Tensor(number, channel, height, width, t1, true, graph);
		
		Tensor b2 = new Tensor(number, channel, height, width, t2, true, graph);
		
		b1.setRequiresGrad(true);
		b2.setRequiresGrad(true);
		
		Tensor c = b1.maximum(b2);
		
		c.showDM();
		
		graph.clearGrad();
		
		graph.backward();
		
		b1.getGrad().showDM();
		
		b2.getGrad().showDM();
		
	}
	
	public static void minimum() {
		
		int number = 1;
		int channel  = 1;
		int height = 1;
		int width = 5;

		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		float[] t1 = new float[] {0.1f,1,0.06f,-1,1.3f};
		
		float[] t2 = new float[] {-0.1f,1,0.07f,-1.2f,0.003f};
		
		Tensor b1 = new Tensor(number, channel, height, width, t1, true, graph);
		
		Tensor b2 = new Tensor(number, channel, height, width, t2, true, graph);
		
		b1.setRequiresGrad(true);
		b2.setRequiresGrad(true);
		
		Tensor c = b1.minimum(b2);
		
		c.showDM();
		
		graph.clearGrad();
		
		graph.backward();
		
		b1.getGrad().showDM();
		
		b2.getGrad().showDM();
		
	}
	
	public static void Lciou() {
		
		float eps = 1e-7f;
		int number = 1;
		int channel  = 4;
		int height = 1;
		int width = 1;
//		int length = number * channel * height * width;

		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		float[] b1a = new float[] {0.5f,0.02f,0.3f,0.6f};
		
		float[] b2a = new float[] {0.3f,0.2f,0.03f,0.12f};
		
		Tensor b1 = new Tensor(number, channel, height, width, b1a, true, graph);
		
		Tensor b2 = new Tensor(number, channel, height, width, b2a, true, graph);
		
		b1.setRequiresGrad(true);
		
		/**
		 * get x y w h
		 */
		Tensor px = b1.get(1, 0, 1);
		Tensor py = b1.get(1, 1, 1);
		Tensor pw = b1.get(1, 2, 1);
		Tensor ph = b1.get(1, 3, 1);
		Tensor pw_= pw.div(2);
		Tensor ph_= ph.div(2);
		
		Tensor tx = b2.get(1, 0, 1);
		Tensor ty = b2.get(1, 1, 1);
		Tensor tw = b2.get(1, 2, 1);
		Tensor th = b2.get(1, 3, 1);
		Tensor tw_ = tw.div(2);
		Tensor th_ = th.div(2);
		
		/**
		 * transform form xywh to xyxy
		 */
		Tensor b1_x1 = px.sub(pw_);
		Tensor b1_x2 = px.add(pw_);
		Tensor b1_y1 = py.sub(ph_);
		Tensor b1_y2 = py.add(ph_);

		Tensor b2_x1 = tx.sub(tw_);
		Tensor b2_x2 = tx.add(tw_);
		Tensor b2_y1 = ty.sub(th_);
		Tensor b2_y2 = ty.add(th_);
		
		/**
		 * Intersection area
		 */
		Tensor iw = b1_x2.minimum(b2_x2).sub(b1_x1.maximum(b2_x1));
		Tensor ih = b1_y2.minimum(b2_y2).sub(b1_y1.maximum(b2_y1));
		Tensor inter = iw.mul(ih);
		
		/**
		 * Union Area
		 * w1 * h1 + w2 * h2 - inter
		 */
		Tensor union = pw.mul(ph).add(tw.mul(th)).sub(inter);
		
		/**
		 * ciou
		 */
		Tensor iou = inter.div(union);

		Tensor cw = b1_x2.maximum(b2_x2).sub(b1_x1.minimum(b2_x1));
		Tensor ch = b1_y2.maximum(b2_y2).sub(b1_y1.minimum(b2_y1));
		Tensor c2 = cw.pow().add(ch.pow());
		Tensor rho2_1 = b2_x1.add(b2_x2).sub(b1_x1).sub(b1_x2).pow(); //(b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
		Tensor rho2_2 = b2_y1.add(b2_y2).sub(b1_y1).sub(b1_y2).pow(); //(b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
		Tensor rho2 = rho2_1.add(rho2_2).div(4);
		float tmp1 = (float) (4.0f / (Math.PI * Math.PI));
		Tensor v = tw.div(th).atan().sub(pw.div(ph).atan()).pow().mul(tmp1);
		Tensor alpha = v.div(v.sub(iou).add(1+eps));
		Tensor ciou = iou.sub(rho2.div(c2).add(v.mul(alpha)));
		
		System.out.println("===================");
		
		ciou.showDM();
		
		graph.clearGrad();
//		graph.showGraph();
		graph.backward();
		
		System.out.println("==========grad=========");
//		t1.getGrad().showDM();
//		alpha.getGrad().showDM();
//		v.getGrad().showDM();
		b1.getGrad().showDM();


	}
	
	public static void atan() {
		
		int number = 1;
		int channel  = 4;
		int height = 1;
		int width = 1;

		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		float[] b1a = new float[] {0.5f,0.02f,0.3f,0.6f};
		
		Tensor b1 = new Tensor(number, channel, height, width, b1a, true, graph);
		b1.setRequiresGrad(true);
		
		Tensor t = b1.atan();
		
		graph.clearGrad();
		graph.backward();
		
		t.showDM();
		b1.getGrad().showDM();
		
	}
	
	public static void silu() {
		
		int number = 1;
		int channel  = 1;
		int height = 1;
		int width = 4;

		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		float[] b1a = new float[] {0.5f,0.02f,0.3f,0.6f};
		
		Tensor x = new Tensor(number, channel, height, width, b1a, true, graph);
		x.setRequiresGrad(true);
		Tensor s = sigmoid(x);
		s.showDM();
		Tensor o = x.mul(s);
		
		graph.clearGrad();
		graph.backward();
		
		o.showDM();
		x.getGrad().showDM();
		
		//output[i] * (1.0f +  x[i] * (1.0f - output[i]))
		// out + sigmoid(x) * (1 - out)
		Tensor d = o.add(s.mul(o.scalarSub(1)));

		d.showDM();	
				
	}
	
	/**
	 * ht = f(W * ht-1 + U * xt + bh)
	 * yt = f(V * ht + by)
	 */
	public static void RNN() {
		
		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		int steps = 3;
		int batchSize = 2;
		int inputSize = 3;
		int hiddenSize = 5;
		
		int number = steps * batchSize;
		
		float[] xd = MatrixUtils.order(steps * batchSize * inputSize, 0.0f, 0.1f);

		Tensor x = new Tensor(number, 1, 1, inputSize, xd, true, graph);
		x.setRequiresGrad(true);
		
//		float[] wd = MatrixUtils.order(inputSize * hiddenSize, 0.0f, 0.1f);
		
		float[] wd = MatrixUtils.val(inputSize * hiddenSize, 0.1f);
		
		Tensor w = new Tensor(1, 1, inputSize, hiddenSize, wd, true, graph);
		w.setRequiresGrad(true);
		
//		float[] ud = MatrixUtils.order(hiddenSize * hiddenSize, 0.0f, 0.2f);
		
		float[] ud = MatrixUtils.val(hiddenSize * hiddenSize, 0.2f);
		
		Tensor u = new Tensor(1, 1, hiddenSize, hiddenSize, ud, true, graph);
		u.setRequiresGrad(true);
		
		float[] vd = MatrixUtils.val(hiddenSize * hiddenSize, 0.01f);
		
		Tensor v = new Tensor(1, 1, hiddenSize, hiddenSize, vd, true, graph);
		v.setRequiresGrad(true);
		
		Tensor out = new Tensor(number, 1, 1, hiddenSize, true, graph);
		
		Tensor h = null;
		
		for(int t = 0;t<steps;t++) {
			
			if(t == 0) {
				
				h = x.get(0, t, batchSize).dot(w);
				
			}else {
				
				h = x.get(0, t, batchSize).dot(w).add(h.dot(u));
				
			}
			
			h = tanh(h);

//			Tensor o = tanh(h.dot(v));

			out.set(h, 0, t * batchSize);
			
		}
		
		graph.clearGrad();
		graph.backward();
		
		System.out.println("x:");
		x.showDM();
		System.out.println("out:");
		out.showDM();
		System.out.println("x-grad:");
		x.getGrad().showDM();
		System.out.println("w-grad:");
		w.getGrad().showDM();
		System.out.println("u-grad:");
		u.getGrad().showDM();
		System.out.println("v-grad:");
		v.getGrad().showDM();
		
	}
	
	public static void selfAttention() {
		
		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		int number = 2;
		int inputSize = 5;
		int hiddenSize = 5;
		
//		float dk =  (float) Math.sqrt(inputSize * 2);
		
		float[] xd = MatrixUtils.order(number * inputSize, 0.0f, 0.1f);
		Tensor x = new Tensor(number, 1, 1, inputSize, xd, true, graph);
		x.setRequiresGrad(true);
		
		float[] qwd = MatrixUtils.val(inputSize * hiddenSize, 0.1f);
		Tensor qw = new Tensor(1, 1, inputSize, hiddenSize, qwd, true, graph);
		qw.setRequiresGrad(true);

		float[] kwd = MatrixUtils.val(inputSize * hiddenSize, 0.2f);
		Tensor kw = new Tensor(1, 1, inputSize, hiddenSize, kwd, true, graph);
		kw.setRequiresGrad(true);
		
		float[] vwd = MatrixUtils.val(inputSize * hiddenSize, 0.01f);
		Tensor vw = new Tensor(1, 1, inputSize, hiddenSize, vwd, true, graph);
		vw.setRequiresGrad(true);
		
		Tensor q = linear(x, qw);
		
		Tensor k = linear(x, kw);
		
		Tensor v = linear(x, vw);
		
//		Tensor aw = q.dot(k.transpose()).div(dk);
		
		Tensor aw = q.dot(k.transpose());
		
		Tensor sf = softmax(aw);

		System.out.println("sf:");
		sf.showDM();
		
		System.out.println("v:");
		v.showDM();
		
		Tensor output = sf.dot(v);
		
		System.out.println("output:");
		output.showDM();
		
		graph.clearGrad();
		graph.backward();
		
//		System.out.println("sf-grad:");
//		sf.getGrad().showDM();
		
		System.out.println("x-grad:");
		x.getGrad().showDM();
		
		System.out.println("qw-grad:");
		qw.getGrad().showDM();
		
		System.out.println("kw-grad:");
		kw.getGrad().showDM();
		
		System.out.println("vw-grad:");
		vw.getGrad().showDM();
		
	}
	
	public static Tensor linear(Tensor x,Tensor w) {
		return x.dot(w);
	}
	
	public static Tensor softmax(Tensor x) {
		Tensor max = x.max(1);
//		System.out.println("x:");
//		x.showDM();
//		x.showShape();
//		System.out.println("max:");
//		max.showDM();
//		max.showShape();
//		System.out.println("sub:");
//		x.sub(max).showDM();
		Tensor e = x.sub(max).exp();
//		System.out.println("e:");
//		e.showDM();
		Tensor sum = e.sum(1);
//		System.out.println("sum:");
//		sum.showShape();
//		sum.showDM();
		return e.div(sum);
	}
	
	public static void softmax_test() {
		
		TensorOP op = new TensorOP(new CUDAManager(0));
		
		Graph graph = new Graph(op);
		
		int number = 2;
		int inputSize = 10;
		
		float[] xd = MatrixUtils.order(number * inputSize, 0.0f, 0.1f);
		Tensor x = new Tensor(number, 1, 1, inputSize, xd, true, graph);
		x.setRequiresGrad(true);
		
		Tensor output = softmax(x);
		
		graph.clearGrad();
		float[] dd = MatrixUtils.order(number * inputSize, 0.0f, 0.1f);
		Tensor delta = new Tensor(number, 1, 1, inputSize, dd, true);
		graph.backward(delta);
		
		output.showDM();
		
		output.getGrad().showDM();
		
	}
	
	public static void main(String[] args) {
		
		try {
			
			CUDAModules.initContext();

			/**
			 * f(x,y)=ln(x)+x*y−sin(y)
			 */
//			formula1();
			
			/**
			 * sigmoid: 1 / 1 + exp(-x)
			 */
//			sigmoid();
			
//			get_gpu();
			
//			int number = 64;
//			int channel  = 125;
//			int height = 32;
//			int width = 32;
//			int length = number * channel * height * width;
//			
//			Tensor x = new Tensor(number, channel, height, width, MatrixUtils.val(length, 0.6f), true);
//			
//			Tensor y = new Tensor(number, channel, height, width, MatrixUtils.val(length, 1f), true);
//			
//			sigmoid_gpu(x, y);
//			
//			x.data = MatrixUtils.val(length, 0.35f);
//			
//			y.data = MatrixUtils.val(length, 2f);
//			
//			Graph.clearGrad();
//			
//			sigmoid_gpu(x, y);
			
//			show();
			
//			pow_gpu();
			
//			multiLabelSoftMarginLoss2();
			
//			sq();
			
//			sum();
			
//			maximum();
			
//			minimum();
			
//			atan();
			
//			Lciou();
			
//			silu();
			
//			RNN();
			
//			selfAttention();
			
			softmax_test();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
			
		}
		
	}

	public TensorOP getTensorOP() {
		return tensorOP;
	}
	
	
}
