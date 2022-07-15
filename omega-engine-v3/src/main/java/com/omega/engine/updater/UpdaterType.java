package com.omega.engine.updater;

/**
 * UpdaterType
 * @author Administrator
 *
 */
public enum UpdaterType {
	
	none("none"),
	momentum("momentum"),
	adam("adam");

	UpdaterType(String key){
		this.key = key;
	}
	
	private String key;

	public String getKey() {
		return key;
	}

}
