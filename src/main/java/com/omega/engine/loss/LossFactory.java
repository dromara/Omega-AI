package com.omega.engine.loss;

import java.util.List;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.YoloLayer;
import com.omega.engine.nn.network.Network;
import com.omega.example.yolo.loss.YoloLoss;
import com.omega.example.yolo.loss.YoloLoss2;
import com.omega.example.yolo.loss.YoloLoss3;
import com.omega.example.yolo.loss.YoloLoss7;

/**
 * LossFactory
 * @author Administrator
 *
 */
public class LossFactory {
	
	/**
	 * create instance
	 * @param type
	 * @return
	 * none null
	 * momentum
	 * adam
	 */
	public static LossFunction create(LossType type,Network network) {
		//square_loss,cross_entropy,softmax_with_cross_entropy
		LossFunction lossFN = null;
		switch (type) {
			case MSE:
				lossFN = new MSELoss(network);
				break;
			case MSE_SUM:
				lossFN = new MSESumLoss(network);
				break;
			case BCE:
				lossFN = new BCELoss(network);
				break;
			case cross_entropy:
				lossFN = new CrossEntropyLoss();
				break;
			case softmax_with_cross_entropy:
				lossFN = new CrossEntropyLoss2(network);
				break;
			case softmax_with_cross_entropy_idx:
				lossFN = new CrossEntropyLossIdx(network);
				break;
			case yolo:
				lossFN = new YoloLoss(1);
				break;
			case yolov3:
				break;
			case yolov7:
				break;
			case multiLabel_soft_margin:
				lossFN = new MultiLabelSoftMargin(network);
				break;
			default:
				break;
		}
		if(lossFN != null && lossFN.getNet() == null) {
			lossFN.setNet(network);
		}
		return lossFN;
	}
	
	/**
	 * create instance
	 * @param type
	 * @return
	 * none null
	 * momentum
	 * adam
	 */
	public static LossFunction create(LossType type,int classNum,Network network) {
		//square_loss,cross_entropy,softmax_with_cross_entropy
		LossFunction lossFN = null;
		switch (type) {
		case MSE:
			lossFN = new MSELoss(network);
			break;
		case BCE:
			lossFN = new BCELoss(network);
			break;
		case cross_entropy:
			lossFN = new CrossEntropyLoss();
			break;
		case softmax_with_cross_entropy:
			lossFN = new CrossEntropyLoss2(network);
			break;
		case softmax_with_cross_entropy_idx:
			lossFN = new CrossEntropyLossIdx(network);
			break;
		case yolo:
			lossFN = new YoloLoss(classNum);
			break;
		case yolov3:
			break;
		case yolov7:
			break;
		case multiLabel_soft_margin:
			lossFN = new MultiLabelSoftMargin(network);
			break;
		default:
			break;
		}
		if(lossFN != null && lossFN.getNet() == null) {
			lossFN.setNet(network);
		}
		return lossFN;
	}
	
	public static LossFunction[] create(LossType type,List<Layer> outputs,Network net) {
		LossFunction[] losses = new LossFunction[outputs.size()];
		
		for(int i = 0;i<outputs.size();i++) {

			switch (type) {
			case MSE:
				losses[i] = new MSELoss(net);
				break;
			case BCE:
				losses[i] = new BCELoss(net);
				break;
			case cross_entropy:
				losses[i] = new CrossEntropyLoss();
				break;
			case softmax_with_cross_entropy:
				losses[i] = new CrossEntropyLoss2(net);
				break;
			case softmax_with_cross_entropy_idx:
				losses[i] = new CrossEntropyLossIdx(net);
				break;
			case yolo:
				losses[i] = new YoloLoss(1);
				break;
			case yolov2:
				YoloLayer yolo2 = (YoloLayer) outputs.get(i);
				losses[i] = new YoloLoss2(yolo2.class_number, yolo2.bbox_num, yolo2.anchors, yolo2.network.getHeight(), yolo2.network.getWidth(), yolo2.maxBox, yolo2.total, yolo2.ignoreThresh, yolo2.truthThresh);
				break;
			case yolov3:
				YoloLayer yolo = (YoloLayer) outputs.get(i);
				losses[i] = new YoloLoss3(yolo.class_number, yolo.bbox_num, yolo.mask, yolo.anchors, yolo.network.getHeight(), yolo.network.getWidth(), yolo.maxBox, yolo.total, yolo.ignoreThresh, yolo.truthThresh);
				break;
			case yolov7:
				YoloLayer yolo7 = (YoloLayer) outputs.get(i);
				losses[i] = new YoloLoss7(yolo7.class_number, yolo7.bbox_num, yolo7.mask, yolo7.anchors, yolo7.network.getHeight(), yolo7.network.getWidth(), yolo7.maxBox, yolo7.total, yolo7.ignoreThresh, yolo7.truthThresh);
				break;
			case multiLabel_soft_margin:
				losses[i] = new MultiLabelSoftMargin(net);
				break;
			default:
				break;
			}
			if(losses[i] != null && losses[i].getNet() == null) {
				losses[i].setNet(net);
			}
		}
		
		return losses;
	}
	
	public static LossFunction[] create(LossType type,List<Layer> outputs,int classNum,Network net) {
		LossFunction[] losses = new LossFunction[outputs.size()];
		
		for(int i = 0;i<outputs.size();i++) {

			switch (type) {
			case MSE:
				losses[i] = new MSELoss(net);
				break;
			case BCE:
				losses[i] = new BCELoss(net);
				break;
			case cross_entropy:
				losses[i] = new CrossEntropyLoss();
				break;
			case softmax_with_cross_entropy:
				losses[i] = new CrossEntropyLoss2(net);
				break;
			case softmax_with_cross_entropy_idx:
				losses[i] = new CrossEntropyLossIdx(net);
				break;
			case yolo:
				losses[i] = new YoloLoss(classNum);
				break;
			case yolov2:
				YoloLayer yolo2 = (YoloLayer) outputs.get(i);
				losses[i] = new YoloLoss2(yolo2.class_number, yolo2.bbox_num, yolo2.anchors, yolo2.network.getHeight(), yolo2.network.getWidth(), yolo2.maxBox, yolo2.total, yolo2.ignoreThresh, yolo2.truthThresh);
				break;
			case yolov3:
				YoloLayer yolo3 = (YoloLayer) outputs.get(i);
				losses[i] = new YoloLoss3(yolo3.class_number, yolo3.bbox_num, yolo3.mask, yolo3.anchors, yolo3.network.getHeight(), yolo3.network.getWidth(), yolo3.maxBox, yolo3.total, yolo3.ignoreThresh, yolo3.truthThresh);
				break;
			case yolov7:
				YoloLayer yolo7 = (YoloLayer) outputs.get(i);
				losses[i] = new YoloLoss7(yolo7.class_number, yolo7.bbox_num, yolo7.mask, yolo7.anchors, yolo7.network.getHeight(), yolo7.network.getWidth(), yolo7.maxBox, yolo7.total, yolo7.ignoreThresh, yolo7.truthThresh);
				break;
			case multiLabel_soft_margin:
				losses[i] = new MultiLabelSoftMargin(net);
				break;
			default:
				break;
			}
			if(losses[i] != null && losses[i].getNet() == null) {
				losses[i].setNet(net);
			}
		}
		
		return losses;
	}
	
}
