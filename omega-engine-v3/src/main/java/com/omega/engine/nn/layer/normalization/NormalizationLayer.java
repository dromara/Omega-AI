package com.omega.engine.nn.layer.normalization;

import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.layer.Layer;

public abstract class NormalizationLayer extends Layer {
	
	public Layer preLayer;
	
	@Override
	public void init() {
		if(preLayer == null) {
			preLayer = this.network.getPreLayer(this.index);
			this.channel = preLayer.oChannel;
			this.height = preLayer.oHeight;
			this.width = preLayer.oWidth;
			this.oChannel = this.channel;
			this.oHeight = this.height;
			this.oWidth = this.width;
		}
		this.number = this.network.number;
		this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
		initParam();
	}
	
	@Override
	public void initBack() {
		this.diff = Blobs.zero(number, channel, height, width, this.diff);
	}
	
}
