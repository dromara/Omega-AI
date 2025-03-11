package com.omega.engine.updater;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.normalization.NormalizationLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.gpu.RMSPropKernel;

/**
 * RMSProp Updater
 * @author Administrator
 *
 */
public class RMSProp extends Updater {

	private RMSPropKernel kernel;
	
	private boolean clamp = false;
	
	private float min = -0.01f;
	
	private float max = 0.01f;
	
	public RMSProp(Network network) {
		this.net = network;
	}
	
	@Override
	public void update(Layer layer) {
		// TODO Auto-generated method stub
		/**
		 * init
		 */
		if(kernel == null) {
			
			if(layer.hasBias) {

				kernel = new RMSPropKernel(layer.weight.dataLength, layer.bias.dataLength, net.cudaManager);
				
			}else {

				kernel = new RMSPropKernel(layer.weight.dataLength, net.cudaManager);
				
			}
			
		}
		
		kernel.updateW(layer.diffW, layer.weight, layer.network, layer.learnRate);
//		
//		System.out.print(layer.getLayerType().toString()+layer.index+":");
//		layer.weight.showDM();

		if(layer.hasBias) {
			kernel.updateB(layer.diffB, layer.bias, layer.network, layer.learnRate);
		}
		
	}

	@Override
	public void updateForMatrix(Layer layer) {
		// TODO Auto-generated method stub

	}

	@Override
	public void updateForBN(NormalizationLayer layer) {
		// TODO Auto-generated method stub
		
//		System.out.println(layer.learnRate);
		/**
		 * init
		 */
		if(kernel == null) {
			kernel = new RMSPropKernel(layer.gamma.dataLength, layer.beta.dataLength, net.cudaManager);
		}

		kernel.updateW(layer.diffGamma, layer.gamma, layer.network, layer.learnRate);
		
		kernel.updateB(layer.diffBeta, layer.beta, layer.network, layer.learnRate);

	}

	@Override
	public UpdaterType getUpdaterType() {
		// TODO Auto-generated method stub
		return UpdaterType.RMSProp;
	}

	@Override
	public void update(Layer layer, int batchSize) {
		// TODO Auto-generated method stub
		/**
		 * init
		 */
		if(kernel == null) {
			
			if(layer.hasBias) {

				kernel = new RMSPropKernel(layer.weight.dataLength, layer.bias.dataLength, net.cudaManager);
				
			}else {

				kernel = new RMSPropKernel(layer.weight.dataLength, net.cudaManager);
				
			}
			
		}
		
		kernel.updateW(layer.diffW, layer.weight, layer.network, layer.learnRate, batchSize);
//		
//		System.out.print(layer.getLayerType().toString()+layer.index+":");
//		layer.weight.showDM();

		if(layer.hasBias) {
			
			kernel.updateB(layer.diffB, layer.bias, layer.network, layer.learnRate, batchSize);
			
		}
		
	}
	
}
