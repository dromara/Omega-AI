package com.omega.utils.fft;

public class Complex {
    public float _r, _i;

    public Complex() {
        reset();
    }

    public Complex(float r) {
        set(r, 0.0f);
    }

    public Complex(float r, float i) {
        set(r, i);
    }

    public Complex(Complex c) {
        set(c);
    }

    public void set(float r, float i) {
        _r = r;
        _i = i;
    }

    public void set(Complex c) {
        _r = c._r;
        _i = c._i;
    }

    public void reset() {
        _r = 0.0f;
        _i = 0.0f;
    }

    public float real() {
        return _r;
    }

    public float imag() {
        return _i;
    }

    public Complex copy() {
        return new Complex(_r, _i);
    }

    public Complex add(float a) {
        return new Complex(_r + a, _i + a);
    }

    public Complex add(Complex a) {
        return new Complex(_r + a._r, _i + a._i);
    }

    public Complex sub(float a) {
        return new Complex(_r - a, _i - a);
    }

    public Complex sub(Complex a) {
        return new Complex(_r - a._r, _i - a._i);
    }

    public Complex mul(float a) {
        return new Complex(a * _r, a * _i);
    }

    public Complex mul(Complex a) {
        return new Complex(_r * a._r - _i * a._i, _i * a._r + _r * a._i);
    }

    public Complex div(float a) {
        return new Complex(_r / a, _i / a);
    }

    public Complex div(Complex a) {
        if (Math.abs(a._r) > Math.abs(a._i)) {
            float s = a._i / a._r;
            float d = 1.0f / (a._r + s * a._i);
            return new Complex(d * (_r + s * _i), d * (_i - s * _r));
        }
        float s = a._r / a._i;
        float d = 1.0f / (a._i + s * a._r);
        return new Complex(d * (s * _r + _i), d * (s * _i - _r));
    }

    public Complex conj() {
        return new Complex(_r, -_i);
    }

    public float abs() {
        if (_r == 0)
            return Math.abs(_i);
        if (_i == 0)
            return Math.abs(_r);
        float ar = Math.abs(_r);
        float ai = Math.abs(_i);
        if (ar > ai) {
            float t = ai / ar;
            return (ar * (float) Math.sqrt(1.0f + t * t));
        }
        float t = ar / ai;
        return (ai * (float) Math.sqrt(1.0f + t * t));
    }

    public Complex exp() {
        float a = (float) Math.exp(_r);
        return new Complex((float) (a * Math.cos(_i)), (float) (a * Math.sin(_i)));
    }

    public float norm() {
        return (_r * _r + _i * _i);
    }
}
