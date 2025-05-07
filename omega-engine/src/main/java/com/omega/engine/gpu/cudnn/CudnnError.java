package com.omega.engine.gpu.cudnn;

public class CudnnError extends RuntimeException {
    /**
     *
     */
    private static final long serialVersionUID = 6948753034479323736L;

    /**
     * Instantiates a new Gpu error.
     *
     * @param message the message
     */
    public CudnnError(final String message) {
        super(message);
    }
}

