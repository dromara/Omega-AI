package com.omega.engine.nn.network;

/**
 * 
 * @author Administrator
 *
 */
public enum NetworkType {
	
	BP("BP"),
	CNN("CNN"),
	ANN("ANN"),
	RNN("RNN");
	
	NetworkType(String key){
		this.key = key;
	}
	
	private String key;

	public String getKey() {
		return key;
	}

}
