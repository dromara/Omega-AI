package com.omega.example.sd.utils;

import java.io.File;
import java.util.List;
import java.util.Map;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAManager;
import com.omega.example.diffusion.utils.DiffusionImageLoader;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.bpe.BPETokenizerEN;
import com.omega.example.unet.utils.SegImageLoader;
import com.omega.example.yolo.data.BaseDataLoader;
import com.omega.example.yolo.data.ImageLoader;
import com.omega.example.yolo.utils.YoloImageUtils;

/**
 * SDImageDataLoaderEN
 * @author Administrator
 *
 */
public class SDImageDataLoaderEN extends BaseDataLoader{
	
	private String labelPath;
	
	private String imgDirPath;
	
	public int img_w;
	
	public int img_h;
	
	private String extName;
	
	private int maxContextLen;
	
	public boolean normalization = true;
	
	private boolean horizontalFilp;

	public float[] mean;
	public float[] std;
	
	public int count;
	
	public int count_it;
	
	private List<Map<String, Object>> datas;
	
	private String[] idxSet;
	
	private BPETokenizerEN tokenizer;
	
	private BaseKernel kernel;
	
	public SDImageDataLoaderEN(BPETokenizerEN tokenizer,String labelPath,String imgDirPath,int img_w,int img_h,int maxContextLen,int batchSize,boolean horizontalFilp) {
		this.horizontalFilp = horizontalFilp;
		this.imgDirPath = imgDirPath;
		this.maxContextLen =  maxContextLen;
		this.tokenizer = tokenizer;
		this.labelPath = labelPath;
		this.img_w = img_w;
		this.img_h = img_h;
		this.batchSize = batchSize;
		init();
	}
	
	public SDImageDataLoaderEN(BPETokenizerEN tokenizer,String labelPath,String imgDirPath,int img_w,int img_h,int maxContextLen,int batchSize,boolean horizontalFilp,float[] mean,float[] std) {
		this.horizontalFilp = horizontalFilp;
		this.imgDirPath = imgDirPath;
		this.labelPath = labelPath;
		this.maxContextLen = maxContextLen;
		this.tokenizer = tokenizer;
		this.img_w = img_w;
		this.img_h = img_h;
		this.batchSize = batchSize;
		this.mean = mean;
		this.std = std;
		init();
	}
	
	public void init() {
		loadFileCount();
	}
	
	public void loadFileCount() {
		
		try {
			
			File file = new File(imgDirPath);
			
			if(file.exists()) {
				datas = LagJsonReader.readJsonDataSamll(labelPath);
				idxSet = new String[datas.size()];
				for(int i = 0;i<datas.size();i++) {
					idxSet[i] = datas.get(i).get("id").toString() + ".png";
				}
			}
			this.number = datas.size();
			count = datas.size();
			count_it = datas.size() / batchSize;
			System.err.println("data count["+count+"].");
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	@Override
	public int[][] shuffle() {
		// TODO Auto-generated method stub
		return MathUtils.randomInts(this.number,this.batchSize);
	}
	
	public int[][] order() {
		// TODO Auto-generated method stub
		return MathUtils.orderInts(this.number,this.batchSize);
	}
	
	@Override
	public void loadData(int pageIndex, int batchSize, Tensor input, Tensor label) {
		// TODO Auto-generated method stub

	}
	
	public void loadData(int pageIndex, int batchSize, Tensor input) {
		// TODO Auto-generated method stub
		
		int[] indexs = getIndexsByAsc(pageIndex, batchSize);
		
		/**
		 * 加载input数据
		 */
		SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, false);
		
		if(normalization) {
			this.normalization(input);
		}
		
		/**
		 * copy data to gpu.
		 */
		input.hostToDevice();
		
	}
	
	@Override
	public float[] loadData(int index) {
		// TODO Auto-generated method stub
		String filePath = imgDirPath + "/" + idxSet[index];
		if(!filePath.contains(".")) {
			filePath += ".png";
		}
		return ImageLoader.resized(filePath, this.img_w, this.img_h);
	}
	
	@Override
	public void loadData(int[] indexs, Tensor input, Tensor label) {
		// TODO Auto-generated method stub
		/**
		 * 加载input数据
		 */
		if(mean != null) {
			SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true, mean , std);
		}else {
			SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true);
		}
		
		loadLabels(indexs, label);
		
		/**
		 * copy data to gpu.
		 */
		input.hostToDevice();
		
		label.hostToDevice();

	}
	
	public void loadData(int[] indexs, Tensor input, Tensor label,Tensor noise) {
		// TODO Auto-generated method stub
		/**
		 * 加载input数据
		 */
		if(mean != null) {
			SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true, mean , std);
		}else {
			SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true);
		}
		
		loadLabels(indexs, label);
		
		RandomUtils.gaussianRandom(noise, 0, 1);
		
		/**
		 * copy data to gpu.
		 */
		input.hostToDevice();
		
		label.hostToDevice();

	}
	
	public void loadData(int[] indexs, Tensor input, Tensor label,Tensor noise,String[] labels) {
		// TODO Auto-generated method stub
		/**
		 * 加载input数据
		 */
		if(mean != null) {
			SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true, mean , std);
		}else {
			SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true);
		}
		
		loadLabels(indexs, label, labels);
		
		RandomUtils.gaussianRandom(noise, 0, 1);
		
		/**
		 * copy data to gpu.
		 */
		input.hostToDevice();
		
		label.hostToDevice();

	}
	
	public void loadData_uncond(int[] indexs, Tensor input, Tensor noise) {
		// TODO Auto-generated method stub
		/**
		 * 加载input数据
		 */
		if(mean != null) {
			SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true, mean , std);
		}else {
			SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true);
		}
		
		RandomUtils.gaussianRandom(noise, 0, 1);
		
		/**
		 * copy data to gpu.
		 */
		input.hostToDevice();

	}
	
	public void loadLabel(int[] indexs, Tensor label) {
		// TODO Auto-generated method stub

		loadLabels(indexs, label);
		
		/**
		 * copy data to gpu.
		 */

		label.hostToDevice();

	}
	
	public void addNoise(Tensor a,Tensor b,Tensor input, Tensor noise,CUDAManager cudaManager) {

		if(kernel == null) {
			kernel = new BaseKernel(cudaManager);
		}
		
		kernel.add_mul(a, b, input, noise, input);
		
	}
	
	public void addNoise(Tensor a,Tensor b,Tensor input, Tensor noise,Tensor output,CUDAManager cudaManager) {

		if(kernel == null) {
			kernel = new BaseKernel(cudaManager);
		}
		
		kernel.add_mul(a, b, input, noise, output);
		
	}
	
	public void unMulGrad(Tensor a,Tensor b,Tensor delta, Tensor noise,Tensor diff,CUDAManager cudaManager) {

		if(kernel == null) {
			kernel = new BaseKernel(cudaManager);
		}
		
		kernel.un_mul_grad(a, b, delta, noise, diff);
		
	}
	
	public void unNoise(Tensor a,Tensor b,Tensor input, Tensor noise,CUDAManager cudaManager) {

		if(kernel == null) {
			kernel = new BaseKernel(cudaManager);
		}
		
		kernel.un_mul(a, b, input, noise, input);
		
	}
	
	public void unNoise(Tensor a,Tensor b,Tensor input, Tensor noise,Tensor output,CUDAManager cudaManager) {

		if(kernel == null) {
			kernel = new BaseKernel(cudaManager);
		}
		
		kernel.un_mul(a, b, input, noise, output);
		
	}
	
	public void loadLabels(int[] indexs,Tensor label) {
		for(int i = 0;i<indexs.length;i++) {
			int idx = indexs[i];
			String text = datas.get(idx).get("en").toString();
			int[] ids = tokenizer.encodeInt(text, maxContextLen);
			for(int j = 0;j<maxContextLen;j++) {
				if(j<ids.length) {
					label.data[i * maxContextLen + j] = ids[j];
				}else {
					label.data[i * maxContextLen + j] = 0;
				}
			}
		}
	}
	
	public void loadLabels(int[] indexs,Tensor label,String[] labels) {
		for(int i = 0;i<indexs.length;i++) {
			int idx = indexs[i];
			String text = datas.get(idx).get("en").toString();
			labels[i] = text;
//			System.out.println(text);
			int[] ids = tokenizer.encodeInt(text, maxContextLen);
			for(int j = 0;j<maxContextLen;j++) {
				if(j<ids.length) {
					label.data[i * maxContextLen + j] = ids[j];
				}else {
					label.data[i * maxContextLen + j] = 0;
				}
			}
		}
//		System.out.println(JsonUtils.toJson(label.data));
	}
	
	public void loadLabels(int[] indexs,Tensor label,Tensor mask,String[] labels) {
		for(int i = 0;i<indexs.length;i++) {
			int idx = indexs[i];
			String text = datas.get(idx).get("en").toString();
			labels[i] = text;
//			System.out.println(text);
			int[] ids = tokenizer.encodeInt(text);
			int[] ids_n = new int[ids.length + 2];
			System.arraycopy(ids, 0, ids_n, 1, ids.length);
			ids_n[0] = tokenizer.sos;
			ids_n[ids_n.length - 1] = tokenizer.eos;
			for(int j = 0;j<maxContextLen;j++) {
				if(j<ids_n.length) {
					label.data[i * maxContextLen + j] = ids_n[j];
					mask.data[i * maxContextLen + j] = 0;
				}else {
					label.data[i * maxContextLen + j] = 0;
					mask.data[i * maxContextLen + j] = -10000.0f;
				}
			}
		}
//		System.out.println(JsonUtils.toJson(label.data));
//		System.out.println(JsonUtils.toJson(mask.data));
	}
	
	public void loadData(int[] indexs,float[] a,float[] b, Tensor input, Tensor noise) {
		// TODO Auto-generated method stub
		RandomUtils.gaussianRandom(noise, 0, 1);

		/**
		 * 加载input数据
		 */
		DiffusionImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, a, b, noise.data, true, horizontalFilp);

		/**
		 * copy data to gpu.
		 */
		input.hostToDevice();

	}
	

	@Override
	public void loadData(int[] indexs, Tensor input) {
		// TODO Auto-generated method stub
		/**
		 * 加载input数据
		 */
		if(mean != null) {
			SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true, mean , std);
		}else {
			SegImageLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, false, true);
		}
		
		/**
		 * copy data to gpu.
		 */
		input.hostToDevice();
	}
	
	public void loadData(String filePath, Tensor input) {
		// TODO Auto-generated method stub
		
		/**
		 * 加载input数据
		 */
		float[] data = YoloImageUtils.loadImgDataToArray(filePath, true);
		System.arraycopy(data, 0, input.data, 0, input.channel * input.height * input.width);
		
		/**
		 * copy data to gpu.
		 */
		input.hostToDevice();
	}
	
	public void loadLabelData(String filePath, Tensor label) {
		// TODO Auto-generated method stub
		
		/**
		 * 加载input数据
		 */
		float[] data = YoloImageUtils.loadImgDataToGrayArray(filePath, true);
		System.arraycopy(data, 0, label.data, 0, label.channel * label.height * label.width);
		
		/**
		 * copy data to gpu.
		 */
		label.hostToDevice();
	}
	
	public void normalization(Tensor input) {
		
		for(int i = 0;i<input.dataLength;i++) {
			int f = (i / input.width / input.height)%input.channel;
			input.data[i] = (input.data[i] - mean[f]) / std[f];
		}
		
	}
	
	public int[] getIndexsByAsc(int pageIndex, int batchSize) {
		
		int start = pageIndex * batchSize;
		
		int end = pageIndex * batchSize + batchSize;
		
		if(end > number) {
			start = start - (end - number);
		}
		
		int[] indexs = new int[batchSize];
		
		for(int i = 0;i<batchSize;i++){
			indexs[i] = start + i;
		}
		return indexs;
	}

	@Override
	public Tensor initLabelTensor() {
		// TODO Auto-generated method stub
		return null;
	}
	
}
