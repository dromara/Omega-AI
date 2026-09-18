package com.omega.example.dit.test;

import java.io.*;
import java.lang.reflect.Field;
import java.nio.file.*;
import java.util.*;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.dit.modules.DiTAttentionLayer2;
import com.omega.engine.nn.layer.gpu.CudnnFlashAttentionKernel;
import com.omega.engine.nn.network.BPNetwork;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

/** Replays captured post-RoPE Q/K/V/dO through the original Java core and FA. */
public class CudnnFlashAttentionSnapshotReplay {
    static Tensor field(DiTAttentionLayer2 layer,String name)throws Exception {
        Field f=DiTAttentionLayer2.class.getDeclaredField(name);f.setAccessible(true);return (Tensor)f.get(layer);
    }
    static void put(Tensor t,float[] x) { JCuda.cudaMemcpy(t.getGpuData(),Pointer.to(x),4L*x.length,cudaMemcpyKind.cudaMemcpyHostToDevice); }
    static float[] get(Tensor t,int n) {float[] x=new float[n];JCuda.cudaMemcpy(Pointer.to(x),t.getGpuData(),4L*n,cudaMemcpyKind.cudaMemcpyDeviceToHost);return x;}
    static float[] rounded(float[] x) {float[] a=x.clone();for(int i=0;i<a.length;i++){int bits=Float.floatToRawIntBits(a[i]);bits+=0x7fff+((bits>>>16)&1);a[i]=Float.intBitsToFloat(bits&0xffff0000);}return a;}
    static void metrics(PrintWriter csv,String group,String name,float[] a,float[] b) {
        double ee=0,bb=0,se=0,max=0,dot=0;int worst=0,bad=0;
        for(int i=0;i<a.length;i++){if(!Float.isFinite(a[i])||!Float.isFinite(b[i])){bad++;continue;}double e=(double)b[i]-a[i];se+=e*e;ee+=(double)a[i]*a[i];bb+=(double)b[i]*b[i];dot+=(double)a[i]*b[i];if(Math.abs(e)>max){max=Math.abs(e);worst=i;}}
        double rel=Math.sqrt(se/Math.max(ee,1e-300)),norm=Math.sqrt(bb/Math.max(ee,1e-300)),cos=dot/Math.sqrt(Math.max(ee*bb,1e-300));
        csv.printf(Locale.ROOT,"%s,%s,%.12g,%.12g,%.12g,%.12g,%d,%d,%.12g,%.12g%n",group,name,rel,max,norm,cos,bad,worst,a[worst],b[worst]);
        System.out.printf(Locale.ROOT,"%s %s relL2=%.9g maxAbs=%.9g normRatio=%.9g nonfinite=%d worst[%d] reference=%.9g test=%.9g%n",group,name,rel,max,norm,bad,worst,a[worst],b[worst]);
    }
    public static void main(String[] args)throws Exception {
        JCuda.setExceptionsEnabled(true);
        Path snapshot=Paths.get(args[0]),report=Paths.get(args[1]);
        int batch,heads,time,dim;float[][] data=new float[4][];
        try(DataInputStream in=new DataInputStream(new BufferedInputStream(Files.newInputStream(snapshot)))){
            if(in.readInt()!=0x4f464131)throw new IOException("Bad OFA1 magic");
            batch=in.readInt();heads=in.readInt();time=in.readInt();dim=in.readInt();
            int n=Math.multiplyExact(Math.multiplyExact(batch,heads),Math.multiplyExact(time,dim));
            if(Files.size(snapshot)!=20L+16L*n)throw new IOException("Snapshot length mismatch");
            for(int j=0;j<4;j++){data[j]=new float[n];for(int i=0;i<n;i++){data[j][i]=in.readFloat();if(!Float.isFinite(data[j][i]))throw new IOException("Nonfinite input");}}
        }
        int repeat=Integer.getInteger("replay.repeatBatch",1);
        if(repeat<1)throw new IllegalArgumentException("repeatBatch must be positive");
        if(repeat>1){for(int j=0;j<4;j++){float[] original=data[j];data[j]=new float[Math.multiplyExact(original.length,repeat)];for(int k=0;k<repeat;k++)System.arraycopy(original,0,data[j],k*original.length,original.length);}batch*=repeat;}
        int n=data[0].length;
        System.out.printf("SNAPSHOT %s B=%d H=%d S=%d D=%d repeatedBatch=%d%n",snapshot,batch,heads,time,dim,repeat);
        for(int j=0;j<4;j++){double sum=0,max=0;for(float x:data[j]){sum+=(double)x*x;max=Math.max(max,Math.abs(x));}System.out.printf(Locale.ROOT,"INPUT %s rms=%.9g maxAbs=%.9g%n",new String[]{"Q","K","V","dO"}[j],Math.sqrt(sum/n),max);}
        attentionStats(data[0],data[1],batch,heads,time,dim);
        BPNetwork network=new BPNetwork(LossType.MSE,UpdaterType.none);network.CUDNN=true;network.CUDNN_SDPA=false;network.RUN_MODEL=RunModel.TRAIN;
        DiTAttentionLayer2 old=new DiTAttentionLayer2(heads*dim,heads,time,true,false,network);
        old.init(new Tensor(batch*time,1,1,heads*dim,true));old.initBack();
        Tensor q=new Tensor(batch,heads,time,dim,data[0],true),k=new Tensor(batch,heads,time,dim,data[1],true);
        Tensor v=field(old,"vt"),temp=field(old,"temp");put(v,data[2]);
        float[][] original=javaCore(old,q,k,v,temp,data[3],n);
        Tensor out=new Tensor(batch,heads,time,dim,true),up=new Tensor(batch,heads,time,dim,data[3],true);
        Tensor dq=new Tensor(batch,heads,time,dim,true),dk=new Tensor(batch,heads,time,dim,true),dv=new Tensor(batch,heads,time,dim,true);
        float[][] flash;
        try(CudnnFlashAttentionKernel fa=new CudnnFlashAttentionKernel(batch,heads,time,dim)){
            fa.forward(q,k,v,out);fa.backward(up,dq,dk,dv);flash=new float[][]{get(out,n),get(dq,n),get(dk,n),get(dv,n)};
        }
        put(q,rounded(data[0]));put(k,rounded(data[1]));put(v,rounded(data[2]));
        float[][] quantized=javaCore(old,q,k,v,temp,rounded(data[3]),n);
        try(PrintWriter csv=new PrintWriter(Files.newBufferedWriter(report,StandardOpenOption.CREATE_NEW))){
            csv.println("case,tensor,relativeL2,maxAbs,normRatio,cosine,nonfinite,worstIndex,reference,test");
            String[] names={"O","dQ","dK","dV"};
            for(int j=0;j<4;j++){metrics(csv,"FA-vs-Java",names[j],original[j],flash[j]);metrics(csv,"roundedJava-vs-Java",names[j],original[j],quantized[j]);metrics(csv,"FA-vs-roundedJava",names[j],quantized[j],flash[j]);}
        }
    }
    static float[][] javaCore(DiTAttentionLayer2 old,Tensor q,Tensor k,Tensor v,Tensor temp,float[] upstream,int n)throws Exception {
        old.scaledDotProductAttention(q,k,v);float[] output=get(temp,n);
        put(temp,upstream);old.scaledDotProductAttentionBackward(q,k);
        return new float[][]{output,get(field(old,"dqt"),n),get(field(old,"dkt"),n),get(field(old,"dvt"),n)};
    }
    static void attentionStats(float[] q,float[] k,int batch,int heads,int time,int dim) {
        // Sampling evenly across queries is sufficient to identify saturated logits without
        // turning snapshot replay into another full CPU attention implementation.
        int queryStep=Math.max(1,time/32);double scale=1.0/Math.sqrt(dim);
        ArrayList<Double> maxProb=new ArrayList<>(),entropy=new ArrayList<>();double largestSpan=0,largestAbs=0;
        for(int b=0;b<batch;b++)for(int h=0;h<heads;h++)for(int qi=0;qi<time;qi+=queryStep){
            int qbase=((b*heads+h)*time+qi)*dim;double max=-Double.MAX_VALUE,min=Double.MAX_VALUE;
            double[] logits=new double[time];
            for(int kj=0;kj<time;kj++){int kbase=((b*heads+h)*time+kj)*dim;double dot=0;for(int d=0;d<dim;d++)dot+=(double)q[qbase+d]*k[kbase+d];dot*=scale;logits[kj]=dot;max=Math.max(max,dot);min=Math.min(min,dot);largestAbs=Math.max(largestAbs,Math.abs(dot));}
            double sum=0;for(double x:logits)sum+=Math.exp(x-max);double e=0,mp=0;for(double x:logits){double p=Math.exp(x-max)/sum;mp=Math.max(mp,p);if(p>0)e-=p*Math.log(p);}
            maxProb.add(mp);entropy.add(e);largestSpan=Math.max(largestSpan,max-min);
        }
        Collections.sort(maxProb);Collections.sort(entropy);int p50=maxProb.size()/2,p99=Math.min(maxProb.size()-1,(int)(maxProb.size()*0.99));
        System.out.printf(Locale.ROOT,"ATTENTION sampledQueries=%d maxAbsLogit=%.9g maxLogitSpan=%.9g maxProbP50=%.9g maxProbP99=%.9g entropyP50=%.9g entropyP01=%.9g%n",maxProb.size(),largestAbs,largestSpan,maxProb.get(p50),maxProb.get(p99),entropy.get(entropy.size()/2),entropy.get(Math.min(entropy.size()-1,(int)(entropy.size()*0.01))));
    }
}
