package com.omega.example.dit.test;

import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.dit.modules.DiTAttentionLayer2;
import com.omega.engine.nn.layer.gpu.CudnnFlashAttentionKernel;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.BPNetwork;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.Pointer;

/** Offline comparison of the actual igone Java training branches, without optimizer updates. */
public final class CudnnFlashAttentionJavaBranchDemo {
    private static final int B = Integer.getInteger("compare.batch", 12);
    private static final int H = Integer.getInteger("compare.heads", 12);
    private static final int S = Integer.getInteger("compare.time", 1101);
    private static final int D = Integer.getInteger("compare.dim", 64);
    private static final int IGNORE = Integer.getInteger("compare.ignore", 77);
    private static final int N = B * H * S * D;
    private static final boolean NORM = Boolean.parseBoolean(System.getProperty("compare.norm", "true"));
    private static final float GAIN = Float.parseFloat(System.getProperty("compare.gain", "1"));
    private static PrintWriter csv;
    private static int failures;

    private static class Probe extends DiTAttentionLayer2 {
        boolean capture = true;
        float[] q, k, v, o, upstream, dq, dk, dv;
        Probe(BPNetwork network) { super(H * D, H, S, true, NORM, network); }
        @Override public void scaledDotProductAttention(Tensor q, Tensor k, Tensor v) {
            if (capture) {
                this.q = snapshot(q, N); this.k = snapshot(k, N); this.v = snapshot(v, N);
            }
            super.scaledDotProductAttention(q, k, v);
            if (capture) o = snapshot(field(this, "temp"), N);
        }
        @Override public void scaledDotProductAttentionBackward(Tensor q, Tensor k) {
            if (capture) upstream = snapshot(field(this, "temp"), N);
            super.scaledDotProductAttentionBackward(q, k);
            // These buffers are overwritten by RoPE/RMSNorm later in diff(..., igone).
            if (capture) {
                dq = snapshot(field(this, "dqt"), N);
                dk = snapshot(field(this, "dkt"), N);
                dv = snapshot(field(this, "dvt"), N);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        int grid = (int) Math.sqrt(S - IGNORE);
        if (grid * grid != S - IGNORE) throw new IllegalArgumentException("S-ignore must be square");
        JCuda.setExceptionsEnabled(true);
        String report = System.getProperty("compare.report", "java-branch.csv");
        if (new java.io.File(report).exists()) throw new IllegalArgumentException("Report already exists");
        csv = new PrintWriter(report, "UTF-8");
        csv.println("case,tensor,status,maxAbs,rmse,relativeL2,cosine,normRatio,nonfinite,worstIndex,javaValue,faValue");
        try {
            System.out.printf("CONFIG B=%d H=%d S=%d D=%d ignore=%d qkNorm=%s gain=%g seed=1234%n", B,H,S,D,IGNORE,NORM,GAIN);
            System.out.println("Layer source: " + DiTAttentionLayer2.class.getProtectionDomain().getCodeSource().getLocation());
            System.out.println("Native library: " + System.getProperty("omega.cudnn.sdpa.library"));
            System.out.println("cuDNN=" + CudnnFlashAttentionKernel.getCudnnVersion());
            run();
            System.out.println("SUMMARY failures=" + failures + " (5% relative RMS/max budget; synthetic inputs only)");
        } finally { csv.close(); }
        if (failures != 0) throw new AssertionError("Numerical budget failures=" + failures);
    }

    private static void run() {
        BPNetwork oldNet = network(false), faNet = network(true);
        Probe old = new Probe(oldNet);
        DiTAttentionLayer2 fa = new DiTAttentionLayer2(H*D,H,S,true,NORM,faNet);
        Random random = new Random(1234);
        FullyLayer[] a = linears(old), b = linears(fa);
        for (int i=0; i<a.length; i++) {
            float[] weights = gaussian(random, H*D*H*D, 1.0f/(float)Math.sqrt(H*D));
            upload(a[i].weight, weights); upload(b[i].weight, weights);
            a[i].bias.clearGPU(); b[i].bias.clearGPU();
        }
        if (NORM) {
            Tensor shape = new Tensor(B,H,S,D,true);
            old.qNorm.init(shape); old.kNorm.init(shape); fa.qNorm.init(shape); fa.kNorm.init(shape);
            float[] gamma = new float[D]; Arrays.fill(gamma, GAIN);
            upload(old.qNorm.gamma,gamma); upload(old.kNorm.gamma,gamma);
            upload(fa.qNorm.gamma,gamma); upload(fa.kNorm.gamma,gamma);
        }
        Tensor input = new Tensor(B*S,1,1,H*D,gaussian(random,N,1),true);
        Tensor delta = new Tensor(B*S,1,1,H*D,gaussian(random,N,0.02f),true);
        Tensor[] rope = RoPEKernel.getCosAndSin2D(S-IGNORE,H*D,H);
        long[] free = new long[1], total = new long[1];
        JCuda.cudaMemGetInfo(free,total);
        System.out.printf("FREE before forward %.1f MiB%n", free[0]/1048576.0);

        old.forward(input,rope[0],rope[1],IGNORE);
        float[] oldOut = snapshot(old.getOutput(),N);
        describe("post-RoPE Q",old.q); describe("post-RoPE K",old.k);
        old.back(delta,rope[0],rope[1],IGNORE);
        Map<String,float[]> oldGrad = gradients(old);

        // Rebuild tensors from snapshots: backward reuses the original layer buffers.
        Tensor q = core(old.q), k = core(old.k), v = core(old.v), out = core(null);
        Tensor upstream = core(old.upstream), dq = core(null), dk = core(null), dv = core(null);
        try (CudnnFlashAttentionKernel sdpa = new CudnnFlashAttentionKernel(B,H,S,D)) {
            sdpa.forward(q,k,v,out); sdpa.backward(upstream,dq,dk,dv);
            compare("core","O",old.o,snapshot(out,N));
            compare("core","dQ",old.dq,snapshot(dq,N));
            compare("core","dK",old.dk,snapshot(dk,N));
            compare("core","dV",old.dv,snapshot(dv,N));
            // Same Java branch on BF16-rounded inputs separates quantization from backend differences.
            Tensor q16=core(roundBf16(old.q)), k16=core(roundBf16(old.k));
            Tensor v16=core(roundBf16(old.v));
            old.capture=false;
            Tensor savedV=field(old,"vt"); setField(old,"vt",v16);
            old.scaledDotProductAttention(q16,k16,v16);
            compare("rounded-java","O",snapshot(field(old,"temp"),N),snapshot(out,N));
            upload(field(old,"temp"),roundBf16(old.upstream));
            old.scaledDotProductAttentionBackward(q16,k16);
            compare("rounded-java","dQ",snapshot(field(old,"dqt"),N),snapshot(dq,N));
            compare("rounded-java","dK",snapshot(field(old,"dkt"),N),snapshot(dk,N));
            compare("rounded-java","dV",snapshot(field(old,"dvt"),N),snapshot(dv,N));
            setField(old,"vt",savedV);
        }

        fa.forward(input,rope[0],rope[1],IGNORE);
        compare("branch-input","Q",old.q,snapshot(field(fa,"rq"),N));
        compare("branch-input","K",old.k,snapshot(field(fa,"rk"),N));
        compare("branch-input","V",old.v,snapshot(field(fa,"vt"),N));
        compare("full-layer","output",oldOut,snapshot(fa.getOutput(),N));
        fa.back(delta,rope[0],rope[1],IGNORE);
        Map<String,float[]> faGrad=gradients(fa);
        for (String name:oldGrad.keySet()) compare("full-layer",name,oldGrad.get(name),faGrad.get(name));

        for (int r=0;r<2;r++) {
            old.forward(input,rope[0],rope[1],IGNORE);
            compare("old-repeat-"+r,"output",oldOut,snapshot(old.getOutput(),N));
            old.back(delta,rope[0],rope[1],IGNORE);
            compare("old-repeat-"+r,"dInput",oldGrad.get("dInput"),snapshot(old.diff,N));
            fa.forward(input,rope[0],rope[1],IGNORE);
            fa.back(delta,rope[0],rope[1],IGNORE);
            compare("fa-repeat-"+r,"dInput",faGrad.get("dInput"),snapshot(fa.diff,N));
        }
        int iterations=Integer.getInteger("compare.iterations",3);
        if(iterations>0) {
            System.out.println("TIMING full-layer forward+backward; includes other GPU process contention; no optimizer");
            for(int round=0;round<3;round++) {
                double oldMs,faMs;
                if(round%2==0) { oldMs=timing(old,input,delta,rope,iterations); faMs=timing(fa,input,delta,rope,iterations); }
                else { faMs=timing(fa,input,delta,rope,iterations); oldMs=timing(old,input,delta,rope,iterations); }
                System.out.printf("TIMING round=%d oldMs=%.4f faMs=%.4f speedup=%.3f%n",round,oldMs,faMs,oldMs/faMs);
            }
        }
        CudnnFlashAttentionKernel plan = (CudnnFlashAttentionKernel) getField(fa,"cudnn_sdpa");
        plan.close();
        // This standalone JVM owns its CUDA allocations; process exit releases all caches.
    }

    private static double timing(DiTAttentionLayer2 layer,Tensor x,Tensor dy,Tensor[] rope,int n) {
        JCuda.cudaDeviceSynchronize(); long start=System.nanoTime();
        for(int i=0;i<n;i++) { layer.forward(x,rope[0],rope[1],IGNORE); layer.back(dy,rope[0],rope[1],IGNORE); }
        JCuda.cudaDeviceSynchronize(); return (System.nanoTime()-start)/1e6/n;
    }
    private static BPNetwork network(boolean flash) {
        BPNetwork net=new BPNetwork(LossType.MSE,UpdaterType.none);
        net.CUDNN=true; net.CUDNN_SDPA=flash; net.RUN_MODEL=RunModel.TRAIN; return net;
    }
    private static FullyLayer[] linears(DiTAttentionLayer2 l) { return new FullyLayer[]{l.qLinerLayer,l.kLinerLayer,l.vLinerLayer,l.oLinerLayer}; }
    private static Map<String,float[]> gradients(DiTAttentionLayer2 l) {
        Map<String,float[]> result=new LinkedHashMap<>(); result.put("dInput",snapshot(l.diff,N));
        String[] names={"q","k","v","o"}; FullyLayer[] layers=linears(l);
        for(int i=0;i<4;i++) {
            result.put(names[i]+".dWeight",snapshot(layers[i].diffW,layers[i].diffW.dataLength));
            result.put(names[i]+".dBias",snapshot(layers[i].diffB,layers[i].diffB.dataLength));
        }
        if(NORM) {
            result.put("q.dGamma",snapshot(l.qNorm.diffGamma,D));
            result.put("k.dGamma",snapshot(l.kNorm.diffGamma,D));
        }
        return result;
    }
    private static Tensor core(float[] values) { return values==null?new Tensor(B,H,S,D,true):new Tensor(B,H,S,D,values,true); }
    private static float[] gaussian(Random r,int n,float scale) { float[] x=new float[n]; for(int i=0;i<n;i++)x[i]=(float)r.nextGaussian()*scale; return x; }
    private static void upload(Tensor t,float[] x) {
        if(x.length>t.dataLength)throw new IllegalArgumentException("Upload overflow");
        JCuda.cudaMemcpy(t.getGpuData(),Pointer.to(x),4L*x.length,cudaMemcpyKind.cudaMemcpyHostToDevice);
    }
    private static float[] snapshot(Tensor t,int n) {
        float[] x=new float[n]; JCuda.cudaMemcpy(Pointer.to(x),t.getGpuData(),4L*n,cudaMemcpyKind.cudaMemcpyDeviceToHost); return x;
    }
    private static Object getField(Object x,String name) {
        try {Field f=DiTAttentionLayer2.class.getDeclaredField(name); f.setAccessible(true); return f.get(x);}
        catch(ReflectiveOperationException e){throw new IllegalStateException(e);}
    }
    private static Tensor field(Object x,String name){return (Tensor)getField(x,name);}
    private static void setField(Object x,String name,Tensor t) {
        try {Field f=DiTAttentionLayer2.class.getDeclaredField(name); f.setAccessible(true);f.set(x,t);}
        catch(ReflectiveOperationException e){throw new IllegalStateException(e);}
    }
    private static float[] roundBf16(float[] input) {
        float[] x=new float[input.length];
        for(int i=0;i<x.length;i++){int bits=Float.floatToRawIntBits(input[i]); bits+=0x7fff+((bits>>>16)&1); x[i]=Float.intBitsToFloat(bits&0xffff0000);}
        return x;
    }
    private static void describe(String name,float[] a) {
        double sum=0,max=0; for(float x:a){sum+=(double)x*x;max=Math.max(max,Math.abs(x));}
        System.out.printf("INPUT %s rms=%.8g maxAbs=%.8g%n",name,Math.sqrt(sum/a.length),max);
    }
    private static void compare(String group,String name,float[] expected,float[] actual) {
        if(expected.length!=actual.length)throw new IllegalArgumentException("Length mismatch");
        double se=0,ee=0,aa=0,dot=0,max=0,refMax=0; int worst=0,nonfinite=0;
        for(int i=0;i<expected.length;i++) {
            double e=expected[i],a=actual[i];
            if(!Double.isFinite(e)||!Double.isFinite(a)){nonfinite++;continue;}
            double diff=a-e; se+=diff*diff; ee+=e*e; aa+=a*a; dot+=e*a; refMax=Math.max(refMax,Math.abs(e));
            if(Math.abs(diff)>max){max=Math.abs(diff);worst=i;}
        }
        double rms=Math.sqrt(se/expected.length), rel=Math.sqrt(se/Math.max(ee,1e-300));
        double cos=ee==0||aa==0?(ee==aa?1:0):dot/Math.sqrt(ee*aa);
        double norm=ee==0?(aa==0?1:Double.POSITIVE_INFINITY):Math.sqrt(aa/ee);
        boolean pass=nonfinite==0 && rms<=1e-6+0.05*Math.sqrt(ee/expected.length) && max<=1e-6+0.05*refMax;
        // Both branches must receive bit-identical Q/K/V, not just meet a numerical budget.
        if(group.equals("branch-input"))pass=nonfinite==0&&max==0;
        if(!pass)failures++;
        String status=pass?"PASS":"FAIL";
        System.out.printf("%s %s/%s relL2=%.9g maxAbs=%.9g normRatio=%.9g nonfinite=%d%n",status,group,name,rel,max,norm,nonfinite);
        System.out.printf("  worst[%d] Java=%.9g FA=%.9g; first Java=%s FA=%s%n",worst,expected[worst],actual[worst],Arrays.toString(Arrays.copyOf(expected,4)),Arrays.toString(Arrays.copyOf(actual,4)));
        csv.printf("%s,%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d,%.12g,%.12g%n",group,name,status,max,rms,rel,cos,norm,nonfinite,worst,expected[worst],actual[worst]);csv.flush();
    }
}
