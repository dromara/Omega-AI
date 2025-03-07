package com.omega.example.transformer.dataset;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.RandomAccessFile;
import java.util.concurrent.CompletableFuture;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BinDataType;
import com.omega.example.transformer.utils.tokenizers.Tokenizer;

import jcuda.runtime.JCuda;

public class SFTBinDataset extends DatasetLoader{
	
	private int batchSize = 1;
	
	private String dataPath;
	
	public int max_len = 256;
	
	public int vocab_size;
	
	public Tensor testInput;
	
	private CompletableFuture<Boolean> cf;
	
	private FileReader fileReader;
	
	private BufferedReader	bufferedReader;
	
	private RandomAccessFile file;
	
	private BinDataType dataType = BinDataType.unint32;
	
	private int index = 0;
	
	private int[] cache = null;
	
	private int[] tmpCount = new int[] {0};
	
	private int byteUnit = 4;
	
	public SFTBinDataset(String dataPath,int max_len,int batchSize,Tokenizer tokenizer,BinDataType dataType) {
		this.dataType = dataType;
		if(dataType == BinDataType.unint16) {
			byteUnit = 2;
		}
		this.dataPath = dataPath;
		this.max_len = max_len;
		this.batchSize = batchSize;
		this.tokenizer = tokenizer;
		this.vocab_size = tokenizer.voc_size();
		this.number = loadBinCount();
		this.count_it = this.number / batchSize;
		System.out.println("dataCount:"+this.number);
		System.out.println("vocab_size:"+this.vocab_size);
		System.out.println("count_it:"+this.count_it);
	}
	
	public void initBinReader() {
		try {
			file.seek(0);
			System.out.println("dataset is ready.");
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}
	
	public int loadBinCount() {
		try {
			file = new RandomAccessFile(getDataPath(), "r");
			number = (int) (file.length() / max_len / byteUnit);
			cache = new int[max_len+1];
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return number;
	}
	
	public void close() {
		try {
			if(bufferedReader!=null) {
				bufferedReader.close();
			}
			if(fileReader!=null) {
				fileReader.close();
			}
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}
	
	public int[] loadData() {
		
		try {

			if((index + 1) * max_len * byteUnit <= file.length()) {
//				System.out.println(index);
				if(getDataType() == BinDataType.unint16) {
					ModelUtils.readShort2Int(file, cache);
				}else {
					ModelUtils.loadIntData(file, cache);
				}
				file.seek(file.getFilePointer() - byteUnit);
				index++;
			}else {
				initBinReader();
				return loadData();
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

		return cache;
	}
	
	public void loadData(Tensor input,Tensor label, float[] tmpInput, float[] tmpLabel,int[] padCount,int it) {
		try {
//			System.out.println(it);
			if(cf != null) {
				boolean success = cf.get();
				if(success) {
					input.hostToDevice(tmpInput);
					label.hostToDevice(tmpLabel);
					System.arraycopy(tmpLabel, 0, label.data, 0, tmpLabel.length);
					padCount[0] = tmpCount[0];
					JCuda.cudaDeviceSynchronize();
				}
				cf = loadAsyncData(tmpInput, tmpLabel, tmpCount);
			}else {
				cf = loadAsyncData(tmpInput, tmpLabel, tmpCount);
			}
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}

	public CompletableFuture<Boolean> loadAsyncData(float[] input,float[] label,int[] counts) {
		CompletableFuture<Boolean> cf = CompletableFuture.supplyAsync(()-> {
			try {
				int number = 0;
				for(int b = 0;b<getBatchSize();b++) {
					int[] onceToken = loadData();
					int count = formatToIdx(b, onceToken, input, label);
					number += count;
				}
				counts[0] = number;
			} catch (Exception e) {
				// TODO: handle exception
				e.printStackTrace();
			}
			return true;
		});
		return cf;
	}
	
	public String decode(Tensor output) {
		int[] tokens = new int[output.number];
		for(int t = 0;t<output.number;t++) {
			int predictIndex = MatrixOperation.maxIndex(output.getByNumber(t));
			tokens[t] = predictIndex;
		}
		return tokenizer.decode(tokens);
	}

	public Tensor loadByTxtToIdx(String txt) {
		int[] idx = tokenizer.encodeInt(txt);
		testInput = Tensor.createTensor(testInput, txt.length(), 1, 1, 1, true);
		for(int t = 0;t<txt.length();t++) {
			testInput.data[t] = idx[t];
		}
		testInput.hostToDevice();
		return testInput;
	}
	
	public Tensor loadByTxtToIdx(int[] idxs) {
//		System.out.println(idxs.length);
		testInput = Tensor.createTensor(testInput, idxs.length, 1, 1, 1, true);
		for(int t = 0;t<idxs.length;t++) {
			testInput.data[t] = idxs[t];
		}
		testInput.hostToDevice();
		return testInput;
	}
	
	public Tensor loadByTxtToIdx(String txt,int maxLen) {
		int[] idx = tokenizer.encodeInt(txt);
		testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
		for(int t = 0;t<idx.length;t++) {
			testInput.data[t] = idx[t];
		}
		testInput.hostToDevice();
		return testInput;
	}
	
	public Tensor loadByTxtToIdx(int[] idxs,int maxLen) {
		if(testInput != null) {
			testInput.clear();
			testInput.clearGPU();
		}
		testInput = Tensor.createTensor(testInput, maxLen, 1, 1, 1, true);
		for(int t = 0;t<idxs.length;t++) {
			testInput.data[t] = idxs[t];
		}
		
		testInput.hostToDevice();
		return testInput;
	}

	public int formatToIdx(int b,int[] onceToken,float[] input,float[] label) {
		int number = 0;
		for(int t = 0;t<max_len;t++) {
			int curr = onceToken[t];
			int next = onceToken[t+1];
			if(next != tokenizer.pad()) {
				number++;
			}
			input[b * max_len + t] = curr;
			label[b * max_len + t] = next;
		}
		return number;
	}

	public static Tensor getPositions(int b,int time) {
		float[] data = new float[b * time];
		for(int n = 0;n<b;n++) {
			for(int t = 0;t<time;t++) {
				data[n * time + t] = t;
			}
		}
		return new Tensor(b * time, 1, 1, 1, data, true);
	}
	
	public static Tensor getPositions(int b,int c,int time) {
		float[] data = new float[b * c * time];
		for(int n = 0;n<b * c;n++) {
			int pt = n % c;
			for(int t = 0;t<time;t++) {
				if(pt == t) {
					data[n * time + t] = 1;
				}
			}
		}
		return new Tensor(b * c, 1, 1, time, data, true);
	}
	
	public static void getPositions(int b,int c,int time,Tensor positions) {
		positions = Tensor.createTensor(positions, b * time, 1, 1, time, true);
		for(int n = 0;n<b * c;n++) {
			int pt = n % b;
			for(int t = 0;t<time;t++) {
				if(pt == t) {
					positions.data[n * time + t] = 1;
				}
			}
		}
		positions.hostToDevice();
	}
	
	public static void getPositions(int b,int time,Tensor positions) {
		positions = Tensor.createTensor(positions, b * time, 1, 1, 1, true);
		for(int n = 0;n<b;n++) {
			for(int t = 0;t<time;t++) {
				positions.data[n * time + t] = t;
			}
		}
		positions.hostToDevice();
	}
	
	public static Tensor triu(int b,int h,int size1,int size2,float val) {
		float[] data = new float[b * h * size1 * size2];
		for(int n = 0;n<b;n++) {
			for(int hn = 0;hn<h;hn++) {
				for(int i = 0;i<size1;i++) {
					for(int j = 0;j<size2;j++) {
						if(i < j) {
							data[n * h * size1 * size2 + hn * size1 * size2 + i * size1 + j] = val;
						}
					}
				}
			}
		}
		
		Tensor mask = new Tensor(b, h, size1, size2, data, true);
		
		return mask;
	}
	
	public static void triu(int b,int h,int size1,int size2,float val,Tensor mask) {
		mask = Tensor.createTensor(mask, b, h, size1, size2, true);
		for(int n = 0;n<b;n++) {
			for(int hn = 0;hn<h;hn++) {
				for(int i = 0;i<size1;i++) {
					for(int j = 0;j<size2;j++) {
						if(i < j) {
							mask.data[n * h * size1 * size2 + hn * size1 * size2 + i * size1 + j] = val;
						}
					}
				}
			}
		}
		mask.hostToDevice();
	}
	
	public static void triu(float val,int[] targetLens,Tensor mask) {
		for(int n = 0;n<mask.number;n++) {
			for(int hn = 0;hn<mask.channel;hn++) {
				for(int i = 0;i<mask.height;i++) {
					for(int j = 0;j<mask.width;j++) {
//						System.out.println(i+":"+targetLens[n]);
						if(i < targetLens[n]) {
							if(i < j) {
//								System.out.println(i+":"+j);
								mask.data[n * mask.channel * mask.height * mask.width + hn * mask.height * mask.width + i * mask.height + j] = val;
							}
						}else {
							mask.data[n * mask.channel * mask.height * mask.width + hn * mask.height * mask.width + i * mask.height + j] = val;
						}
					}
				}
			}
		}
		mask.hostToDevice();
	}

	public BinDataType getDataType() {
		return dataType;
	}

	public String getDataPath() {
		return dataPath;
	}

	public int getBatchSize() {
		return batchSize;
	}
	
}
