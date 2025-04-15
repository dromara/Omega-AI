package com.omega.utils.fft;

import com.omega.utils.JsonUtils;

public final class RealFFT {
    private int _nt, _nf;
    private float[] _table;
    private float[] _a, _aw;
    private float c1, c2, c3, c4;
    private float s1, s2, s3, s4;
    private int _nfft, _nfft2;

    public RealFFT() {
    }

    static public int nfft(int n) {
        int nfft = n;
        boolean next = true;
        while (next) {
            // Test for factor of 2
            if (n % 2 == 0) {
                n /= 2;
                if (n == 1)
                    next = false;
            }
            // Test for factor of 3
            else if (n % 3 == 0) {
                n /= 3;
                if (n == 1)
                    next = false;
            }
            // Test for factor of 5
            else if (n % 5 == 0) {
                n /= 5;
                if (n == 1)
                    next = false;
            }
            // All factors failed, increase test by 1
            else {
                nfft++;
                n = nfft;
            }
        }
        return nfft;
    }

    public static void main(String[] args) {
        float[] realInput = new float[]{94, 94, 112, 112, 112, 94, 94, 94};
        RealFFT fft = new RealFFT();
        long start = System.nanoTime();
        Complex[] complexOutput = fft.forward(realInput);
        System.out.println("fft run time: " + (System.nanoTime() - start) / 1e6 + "ms");
        System.out.println(JsonUtils.toJson(complexOutput));
        float[] realOutput = fft.inverse(complexOutput);
        System.out.println(JsonUtils.toJson(realOutput));
        float[] abs = new float[complexOutput.length];
        for (int i = 0; i < complexOutput.length; i++) {
            Complex com = complexOutput[i];
            abs[i] = com.abs();
        }
        System.out.println(JsonUtils.toJson(abs));
        float err = 0.0f;
        for (int i = 0; i < realOutput.length; i++) {
            err += Math.abs(realInput[i] - realOutput[i]);
        }
        System.out.println(" Total   absolute error: " + err);
        System.out.println(" Average absolute error: " + (err / realOutput.length));
    }

    public Complex[] forward(float[] r) {
        _nt = r.length;
        _nf = _nt / 2 + 1;
        checkTable();
        _a = new float[_nt];
        for (int i = 0; i < _nt; i++) {
            _a[i] = r[i];
        }
        fft(1);
        Complex[] c = new Complex[_nf];
        c[0] = new Complex(_a[0]);
        c[_nf - 1] = new Complex(_a[1]);
        for (int i = 1; i < _nf - 1; i++) {
            c[i] = new Complex(_a[2 * i], _a[2 * i + 1]);
        }
        return c;
    }

    public float[] abs(Complex[] c) {
        float[] abs = new float[c.length];
        for (int i = 0; i < c.length; i++) {
            Complex com = c[i];
            abs[i] = com.abs();
        }
        return abs;
    }

    public float[][] batchForwardABS(float[][] r) {
        int nt = r[0].length;
        nt = nt / 2 + 1;
        float[][] batch = new float[r.length][nt];
        for (int b = 0; b < r.length; b++) {
            batch[b] = abs(forward(r[b]));
        }
        return batch;
    }

    public float[] inverse(Complex[] c) {
        _nf = c.length;
        _nt = 2 * (_nf - 1);
        checkTable();
        _a = new float[_nt];
        _a[0] = c[0]._r;
        _a[1] = c[_nf - 1]._r;
        int j = 2;
        for (int i = 1; i < _nf - 1; i++) {
            _a[j++] = c[i]._r;
            _a[j++] = c[i]._i;
        }
        fft(-1);
        float scale = 2.0f / _nt;
        for (int i = 0; i < _nt; i++) {
            _a[i] *= scale;
        }
        return _a;
    }

    private void computeTable() {
        _table = new float[_nf];
        _table[0] = 1.0f;
        _table[1] = 0.0f;
        double f = -Math.PI / (_nf - 1);
        for (int i = 2; i < _nf - 1; i += 2) {
            double ff = i * f;
            _table[i] = (float) Math.cos(ff);
            _table[i + 1] = (float) Math.sin(ff);
        }
    }

    private void checkTable() {
        if (_table == null) {
            computeTable();
        } else {
            if (_table.length != _nf)
                computeTable();
        }
    }

    private void applyTable(int dir) {
        int nm1 = _nf - 1;
        double f = Math.PI / nm1;
        float fcos = (float) Math.cos(f);
        float fsin = (float) Math.sin(f);
        float t = _a[0] + _a[1];
        _a[1] = _a[0] - _a[1];
        _a[0] = t;
        if (dir < 0) {
            _a[0] = 0.5f * _a[0];
            _a[1] = 0.5f * _a[1];
        }
        int n = 2 * _nf - 2;
        for (int i = 2; i < nm1; i += 2) {
            int j = n - i;
            int jp1 = j + 1;
            int ip1 = i + 1;
            float a1 = 0.5f * (_a[i] + _a[j]);
            float a2 = 0.5f * (_a[ip1] + _a[jp1]);
            float a3 = 0.5f * (_a[i] - _a[j]);
            float a4 = 0.5f * (_a[ip1] - _a[jp1]);
            int iodd = i / 2;
            int ieve = 2 * (iodd / 2);
            iodd = iodd - ieve;
            float acos = _table[ieve++];
            float asin = -_table[ieve];
            if (iodd == 0) {
                iodd = 1;
            } else {
                t = acos * fcos - asin * fsin;
                asin = asin * fcos + acos * fsin;
                acos = t;
                iodd = 0;
            }
            float a5 = a3 * asin;
            float a6 = a2 * asin;
            a2 *= acos;
            a3 *= acos;
            if (dir < 0) {
                a2 = -a2;
                a3 = -a3;
            }
            _a[i] = a1 - a5 + a2;
            _a[i + 1] = a4 - a6 - a3;
            _a[j] = a1 + a5 - a2;
            _a[j + 1] = -a4 - a6 - a3;
        }
        if (nm1 == (nm1 / 2) * 2)
            _a[_nf] = -_a[_nf];
    }

    private void fft(int dir) {
        _nfft = _a.length / 2;
        if (_nfft < 2)
            return;
        _nfft2 = 2 * _nfft;
        _aw = new float[_nt];
        c1 = 0.0f;
        c2 = 0.0f;
        c3 = 0.0f;
        c4 = 0.0f;
        s1 = 0.0f;
        s2 = 0.0f;
        s3 = 0.0f;
        s4 = 0.0f;
        // For inverse fft apply cosine/sines to input
        if (dir < 0)
            applyTable(dir);
        // Iterate through input (use factors 2,3,4,5 as possible)
        int now = _nfft;
        int lold = 2, lnew = 0;
        boolean nextFactor = true;
        while (nextFactor) {
            // Test for odd factor of 2 or factor of 4 (even power of 2)
            if (now % 2 == 0) {
                int pow = 1;
                int rem = now / 2;
                int ip = 0;
                while (rem > 0 && ip == 0) {
                    if (rem % 2 == 0) {
                        pow++;
                        rem /= 2;
                    } else {
                        if (pow % 2 == 1) {
                            now /= 2;
                            ip = 2;
                            lnew = factor2();
                        } else {
                            now /= 4;
                            ip = 4;
                            lnew = factor4(lold, dir);
                        }
                    }
                }
            }
            // Test for factor of 4
            //		      if ( now%4 == 0 ) {
            //		        now /= 4;
            //		        lnew = factor4( lold, dir );
            //		      }
            // Test for factor of 2
            //		      else if ( now%2 == 0 ) {
            //		        now /= 2;
            //		        lnew = factor2();
            //		      }
            // Test for factor of 3
            else if (now % 3 == 0) {
                now /= 3;
                lnew = factor3(lold, dir);
            }
            // Test for factor of 5
            else if (now % 5 == 0) {
                now /= 5;
                lnew = factor5(lold, dir);
            }
            // General factors remaining
            else {
                lnew = factorN(now, lold, dir);
            }
            for (int i = 0; i < _nfft2; i++) {
                _a[i] = _aw[i];
                i++;
                _a[i] = _aw[i];
            }
            if (lnew == _nfft2)
                nextFactor = false;
            lold = lnew;
        }
        // For forward fft apply cosine/sines to result
        if (dir > 0)
            applyTable(dir);
    }

    private int factor2() {
        int k = 0;
        for (int i = 0; i < _nfft; i += 2) {
            int j = i + _nfft, ip1 = i + 1, jp1 = j + 1;
            _aw[k++] = _a[i] + _a[j];
            _aw[k++] = _a[ip1] + _a[jp1];
            _aw[k++] = _a[i] - _a[j];
            _aw[k++] = _a[ip1] - _a[jp1];
        }
        return 4;
    }

    private int factor4(int lold, int dir) {
        float ar, ai, tr0, tr1, tr2, tr3, ti0, ti1, ti2, ti3;
        int ns = _nfft2 / 4;
        int lnew = lold * 4;
        int ndel = (2 * _nfft2) / lnew;
        int nw = 0;
        for (int i = 0; i < lold; i += 2) {
            int k0 = i;
            if (i != 0) {
                nw += ndel;
                c1 = _table[nw];
                s1 = _table[nw + 1];
                int jj = nw + nw;
                c2 = _table[jj++];
                s2 = _table[jj];
                if (dir < 0) {
                    s1 = -s1;
                    s2 = -s2;
                }
                c3 = c1 * c2 - s1 * s2;
                s3 = c1 * s2 + s1 * c2;
            }
            for (int k = 1; k <= ns; k += lold) {
                int k1 = k0 + lold;
                int k2 = k1 + lold;
                int k3 = k2 + lold;
                int j0 = k + i;
                int j0m1 = j0 - 1;
                int j1 = j0 + ns;
                int j1m1 = j1 - 1;
                int j2 = j1 + ns;
                int j2m1 = j2 - 1;
                int j3 = j2 + ns;
                int j3m1 = j3 - 1;
                if (i != 0) {
                    tr1 = c1 * _a[j1m1] - s1 * _a[j1];
                    ti1 = c1 * _a[j1] + s1 * _a[j1m1];
                    tr2 = c2 * _a[j2m1] - s2 * _a[j2];
                    ti2 = c2 * _a[j2] + s2 * _a[j2m1];
                    tr3 = c3 * _a[j3m1] - s3 * _a[j3];
                    ti3 = c3 * _a[j3] + s3 * _a[j3m1];
                    tr0 = _a[j0m1] + tr2;
                    ti0 = _a[j0] + ti2;
                    tr2 = _a[j0m1] - tr2;
                    ti2 = _a[j0] - ti2;
                    ar = tr1 + tr3;
                    ai = ti1 + ti3;
                    tr3 = tr1 - tr3;
                    ti3 = ti1 - ti3;
                } else {
                    tr0 = _a[j0m1] + _a[j2m1];
                    ti0 = _a[j0] + _a[j2];
                    tr2 = _a[j0m1] - _a[j2m1];
                    ti2 = _a[j0] - _a[j2];
                    ar = _a[j1m1] + _a[j3m1];
                    ai = _a[j1] + _a[j3];
                    tr3 = _a[j1m1] - _a[j3m1];
                    ti3 = _a[j1] - _a[j3];
                }
                if (dir < 0) {
                    tr3 = -tr3;
                    ti3 = -ti3;
                }
                _aw[k0] = tr0 + ar;
                _aw[k0 + 1] = ti0 + ai;
                _aw[k1++] = tr2 + ti3;
                _aw[k1] = ti2 - tr3;
                _aw[k2++] = tr0 - ar;
                _aw[k2] = ti0 - ai;
                _aw[k3++] = tr2 - ti3;
                _aw[k3] = ti2 + tr3;
                k0 += lnew;
            }
        }
        return lnew;
    }

    private int factor3(int lold, int dir) {
        float wc = -0.5000000f;
        float ws = -0.8660300f;
        if (dir < 0)
            ws = -ws;
        float ar, ai, tr0, tr1, tr2, ti0, ti1, ti2;
        int ns = _nfft2 / 3;
        int lnew = 3 * lold;
        int ndel = (2 * _nfft2) / lnew;
        int nw = 0;
        for (int i = 0; i < lold; i += 2) {
            int k0 = i;
            if (i != 0) {
                nw += ndel;
                c1 = _table[nw];
                s1 = _table[nw + 1];
                if (dir < 0)
                    s1 = -s1;
                c2 = s1 * s1;
                c2 = 1.0f - c2 - c2;
                s2 = s1 * c1;
                s2 += s2;
            }
            for (int k = 1; k <= ns; k += lold) {
                int k1 = k0 + lold;
                int k2 = k1 + lold;
                int j0 = k + i;
                int j0m1 = j0 - 1;
                int j1 = j0 + ns;
                int j1m1 = j1 - 1;
                int j2 = j1 + ns;
                int j2m1 = j2 - 1;
                if (i != 0) {
                    tr1 = c1 * _a[j1m1] - s1 * _a[j1];
                    ti1 = c1 * _a[j1] + s1 * _a[j1m1];
                    tr2 = c2 * _a[j2m1] - s2 * _a[j2];
                    ti2 = c2 * _a[j2] + s2 * _a[j2m1];
                    tr0 = tr1 + tr2;
                    ti0 = ti1 + ti2;
                    tr1 = tr1 - tr2;
                    ti1 = ti1 - ti2;
                } else {
                    tr0 = _a[j1m1] + _a[j2m1];
                    ti0 = _a[j1] + _a[j2];
                    tr1 = _a[j1m1] - _a[j2m1];
                    ti1 = _a[j1] - _a[j2];
                }
                tr2 = tr0 * wc;
                ti2 = ti0 * wc;
                ar = ti1 * ws;
                ai = tr1 * ws;
                _aw[k0] = _a[j0m1] + tr0;
                _aw[k0 + 1] = _a[j0] + ti0;
                tr0 = _a[j0m1] + tr2;
                ti0 = _a[j0] + ti2;
                _aw[k1++] = tr0 - ar;
                _aw[k1] = ti0 + ai;
                _aw[k2++] = tr0 + ar;
                _aw[k2] = ti0 - ai;
                k0 += lnew;
            }
        }
        return lnew;
    }

    private int factor5(int lold, int dir) {
        float wc1 = 0.3090170f, wc2 = -0.8090170f;
        float ws1 = -0.9510565f, ws2 = -0.5877852f;
        if (dir < 0) {
            ws1 = -ws1;
            ws2 = -ws2;
        }
        float tr0, ti0, tr1, ti1, tr2, ti2, tr3, ti3, tr4, ti4;
        int ns = _nfft2 / 5;
        int lnew = 5 * lold;
        int ndel = (2 * _nfft2) / lnew;
        int nw = 0;
        for (int i = 0; i < lold; i += 2) {
            int k0 = i;
            if (i != 0) {
                nw += ndel;
                c1 = _table[nw];
                s1 = _table[nw + 1];
                int jj = nw + nw;
                c2 = _table[jj++];
                s2 = _table[jj];
                if (dir < 0) {
                    s1 = -s1;
                    s2 = -s2;
                }
                c3 = c1 * c2 - s1 * s2;
                s3 = c1 * s2 + c2 * s1;
                s4 = s2 * c2;
                s4 = s4 + s4;
                c4 = s2 * s2;
                c4 = 1.0f - c4 - c4;
            }
            for (int k = 1; k <= ns; k += lold) {
                int k1 = k0 + lold;
                int k2 = k1 + lold;
                int k3 = k2 + lold;
                int k4 = k3 + lold;
                int j0 = k + i;
                int j0m1 = j0 - 1;
                int j1 = j0 + ns;
                int j1m1 = j1 - 1;
                int j2 = j1 + ns;
                int j2m1 = j2 - 1;
                int j3 = j2 + ns;
                int j3m1 = j3 - 1;
                int j4 = j3 + ns;
                int j4m1 = j4 - 1;
                if (i != 0) {
                    tr1 = c1 * _a[j1m1] - s1 * _a[j1];
                    ti1 = c1 * _a[j1] + s1 * _a[j1m1];
                    tr2 = c2 * _a[j2m1] - s2 * _a[j2];
                    ti2 = c2 * _a[j2] + s2 * _a[j2m1];
                    tr3 = c3 * _a[j3m1] - s3 * _a[j3];
                    ti3 = c3 * _a[j3] + s3 * _a[j3m1];
                    tr4 = c4 * _a[j4m1] - s4 * _a[j4];
                    ti4 = c4 * _a[j4] + s4 * _a[j4m1];
                    tr0 = tr1 + tr4;
                    ti0 = ti1 + ti4;
                    tr1 = tr1 - tr4;
                    ti1 = ti1 - ti4;
                    tr4 = tr2 + tr3;
                    ti4 = ti2 + ti3;
                    tr3 = tr2 - tr3;
                    ti3 = ti2 - ti3;
                } else {
                    tr0 = _a[j1m1] + _a[j4m1];
                    ti0 = _a[j1] + _a[j4];
                    tr1 = _a[j1m1] - _a[j4m1];
                    ti1 = _a[j1] - _a[j4];
                    tr4 = _a[j2m1] + _a[j3m1];
                    ti4 = _a[j2] + _a[j3];
                    tr3 = _a[j2m1] - _a[j3m1];
                    ti3 = _a[j2] - _a[j3];
                }
                float ar = tr0 * wc1 + tr4 * wc2 + _a[j0m1];
                float ai = ti0 * wc1 + ti4 * wc2 + _a[j0];
                float br = tr0 * wc2 + tr4 * wc1 + _a[j0m1];
                float bi = ti0 * wc2 + ti4 * wc1 + _a[j0];
                float cr = ti1 * ws1 + ti3 * ws2;
                float ci = tr1 * ws1 + tr3 * ws2;
                float dr = ti1 * ws2 - ti3 * ws1;
                float di = tr1 * ws2 - tr3 * ws1;
                _aw[k0] = _a[j0m1] + tr0 + tr4;
                _aw[k0 + 1] = _a[j0] + ti0 + ti4;
                _aw[k1++] = ar - cr;
                _aw[k1] = ai + ci;
                _aw[k2++] = br - dr;
                _aw[k2] = bi + di;
                _aw[k3++] = br + dr;
                _aw[k3] = bi - di;
                _aw[k4++] = ar + cr;
                _aw[k4] = ai - ci;
                k0 += lnew;
            }
        }
        return lnew;
    }

    private int factorN(int n, int lold, int dir) {
        int ns = _nfft2 / n;
        int ns1 = ns + 1;
        int ns2 = (_nfft2 + ns) / 2;
        int lnew = n * lold;
        int ndel = (2 * _nfft2) / lnew;
        if (lold != 2) {
            _aw[0] = 1.0f;
            _aw[1] = 0.0f;
            _aw[_nfft] = -1.0f;
            _aw[_nfft + 1] = 0.0f;
            for (int i = 2; i < _nfft; i += 2) {
                int ip1 = i + 1;
                int nmi = _nfft2 - i;
                _aw[i] = _table[i];
                _aw[nmi] = _table[i];
                _aw[ip1] = _table[ip1];
                if (dir < 0)
                    _aw[ip1] = -_aw[ip1];
                _aw[nmi + 1] = -_aw[ip1];
            }
            int ndw = 0;
            for (int k = ns1; k < _nfft2; k += ns) {
                ndw += ndel;
                int nw = 0;
                for (int i = 2; i < lold; i += 2) {
                    nw += ndw;
                    int nw1 = nw + 1;
                    int ipk = i + k;
                    for (int j = 0; j < ns; j += lold) {
                        int jj = j + ipk;
                        int jjj = jj - 1;
                        float t = _a[jjj] * _aw[nw] - _a[jj] * _aw[nw1];
                        _a[jj] = _a[jj] * _aw[nw] + _a[jjj] * _aw[nw1];
                        _a[jjj] = t;
                    }
                }
            }
        }
        int jj = 0;
        for (int j = 0; j < ns; j += lold) {
            for (int i = 0; i < lold; i += 2) {
                int ij = i + j;
                int ijj = i + jj;
                _aw[ijj] = _a[ij];
                _aw[ijj + 1] = _a[ij + 1];
                for (int k = ns1; k < _nfft2; k += ns) {
                    int ijk = ij + k;
                    _aw[ijj] += _a[ijk - 1];
                    _aw[ijj + 1] += _a[ijk];
                }
            }
            jj += lnew;
        }
        ndel = 0;
        int ll = lold;
        for (int l = ll; l < (lnew + lold) / 2; l += lold) {
            ndel += ns;
            int nw = 0;
            int lc = lnew - 2 * (l + 1);
            for (int k = 0; k < ns2; k += ns) {
                int kk = _nfft2 - k - 2;
                if (k != 0) {
                    nw += ndel;
                    if (nw > _nfft2)
                        nw -= _nfft2;
                    int nw1 = nw + 1;
                    if (nw <= _nfft) {
                        c1 = _table[nw];
                        s1 = _table[nw1];
                    } else {
                        int kkk = _nfft2 - nw;
                        c1 = _table[kkk++];
                        s1 = -_table[kkk];
                    }
                    if (dir < 0)
                        s1 = -s1;
                }
                ll = l;
                for (int j = 0; j < ns; j += lold) {
                    int j1 = j + k;
                    int j2 = j + kk + 1;
                    for (int i = 1; i <= lold; i += 2) {
                        int k1 = i + ll;
                        int k2 = k1 + lc + 1;
                        int j3 = i + j1;
                        int j3m1 = j3 - 1;
                        int j4 = i + j2;
                        int j4p1 = j4 + 1;
                        if (k <= 0) {
                            _aw[k1 - 1] = _a[j3m1];
                            _aw[k1] = _a[j3];
                            _aw[k2++] = _a[j3m1];
                            _aw[k2] = _a[j3];
                        } else {
                            float tr1 = c1 * (_a[j4] + _a[j3m1]);
                            float ti1 = c1 * (_a[j4p1] + _a[j3]);
                            float tr2 = s1 * (_a[j3] - _a[j4p1]);
                            float ti2 = s1 * (_a[j3m1] - _a[j4]);
                            _aw[k1 - 1] += (tr1 - tr2);
                            _aw[k1] += (ti1 + ti2);
                            _aw[k2++] += (tr1 + tr2);
                            _aw[k2] += (ti1 - ti2);
                        }
                    }
                    ll += lnew;
                }
            }
        }
        return lnew;
    }
}
