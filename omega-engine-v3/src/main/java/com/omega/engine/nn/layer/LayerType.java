package com.omega.engine.nn.layer;

/**
 * layer type
 * @author Administrator
 * 
 */
public enum LayerType {
	
	full("full"),
	softmax("softmax"),
	conv("conv"),
	pooling("pooling"),
	input("input"),
	softmax_cross_entropy("softmax_cross_entropy"),
	sigmod("sigmod"),
	relu("relu"),
	tanh("tanh"),
	bn("bn"),
	block("block"),
	dropout("dropout");
	
	private String key;
	
	LayerType(String key){
		this.key = key;
	}

	public String getKey() {
		return key;
	}
	
	public static LayerType getEnumByKey(String key){
		for(LayerType type:LayerType.values()) {
			if(type.getKey().equals(key)) {
				return type;
			}
		}
		return null;
	}

}
