package com.omega.engine.nn.layer.active;

import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.layer.Layer;

public abstract class ActiveFunctionLayer extends Layer {
	
	@Override
	public void init() {
		Layer preLayer = this.network.getPreLayer(this.index);
		if(this.parent != null) {
			preLayer = this.parent.layers.get(index - 1);
		}
		this.number = preLayer.number;
		this.channel = preLayer.oChannel;
		this.height = preLayer.oHeight;
		this.width = preLayer.oWidth;
		this.oChannel = this.channel;
		this.oHeight = this.height;
		this.oWidth = this.width;
		this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
	}
	
	@Override
	public void initBack() {
		this.diff = Blobs.zero(number, channel, height, width, this.diff);
	}
	
}
