package com.omega.engine.check;

import com.omega.engine.tensor.Tensor;

public abstract class BaseCheck {
    public abstract float check(Tensor output, Tensor label, String[] labelSet, boolean showErrorLabel);
}

