package com.omega.engine.nn.network;

public enum RunModel {
	
	TRAIN("train"),
	EVAL("eval"),
	TEST("test");
	
	RunModel(String key){
		this.key = key;
	}
	
	private String key;

	public String getKey() {
		return key;
	}
	
}
