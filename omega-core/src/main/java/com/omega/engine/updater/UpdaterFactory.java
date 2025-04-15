package com.omega.engine.updater;

import com.omega.engine.nn.network.Network;

/**
 * Updater Factory
 *
 * @author Administrator
 * <p>
 * none
 * <p>
 * momentum
 * <p>
 * adam
 */
public class UpdaterFactory {
    /**
     * create instance
     *
     * @param type
     * @return none null
     * <p>
     * momentum
     * <p>
     * adam
     */
    public static Updater create(Network network) {
        switch (network.updater) {
            case momentum:
                return new Momentum(network);
            case sgd:
                return new SGDM(network);
            case adam:
                return new Adam(network);
            case adamw:
                return new AdamW(network);
            case RMSProp:
                return new RMSProp(network);
            default:
                return null;
        }
    }
}

