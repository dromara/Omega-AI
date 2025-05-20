package com.omega.engine.nn.layer;

import com.omega.engine.tensor.Tensor;

public abstract class BaseRNNLayer extends Layer {
    public int layerNum = 1;
    public int rnnMode = 1;  //rnn cell + tanh

    public abstract void forward(int time, int number);

    public abstract void forward(Tensor input, Tensor hx, Tensor cx, int time);

    public abstract void back(Tensor delta, Tensor hx, Tensor cx, Tensor dhy, Tensor dcy);

    public abstract Tensor getHx();

    public abstract Tensor getDhx();

    public abstract Tensor getCx();

    public abstract Tensor getDcx();

    public abstract Tensor getHy();

    public abstract Tensor getDhy();

    public abstract Tensor getCy();

    public abstract Tensor getDcy();
}

