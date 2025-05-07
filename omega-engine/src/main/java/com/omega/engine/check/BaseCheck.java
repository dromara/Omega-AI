package com.omega.engine.check;

import com.omega.common.data.Tensor;

public abstract class BaseCheck {
    public abstract float check(Tensor output, Tensor label, String[] labelSet, boolean showErrorLabel);
}

