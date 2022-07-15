package com.omega.engine.updater;

/**
 * Updater Factory
 * @author Administrator
 * none
 * momentum
 * adam
 */
public class UpdaterFactory {
	
	/**
	 * create instance
	 * @param type
	 * @return
	 * none null
	 * momentum
	 * adam
	 */
	public static Updater create(UpdaterType type) {
		
		switch (type) {
		case momentum:
			return new Momentum();
		case adam:
			return new Adam();
		default:
			return null;
		}
		
	}
	
	
}
