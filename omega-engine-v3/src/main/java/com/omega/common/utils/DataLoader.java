package com.omega.common.utils;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.data.MatrixDataSet;

/**
 * data load utils
 * @author Administrator
 *
 */
public class DataLoader {
	
	/**
	 * loalDataByTxt(labelSize is 1)
	 * @param filePath
	 * @param breakSign
	 * @param inputSize
	 * @param labelSize
	 * @return
	 */
	public static DataSet loalDataByTxt(String filePath,String breakSign,int channel,int height,int width,int labelSize) {
		

		InputStreamReader read = null;
		
		FileInputStream fio = null;
		
		try {
			
			if(filePath == null || filePath.equals("")) {
				throw new RuntimeException("filePath is null!");
			}
			
			File file = new File(filePath);
			
			if(!file.exists()) {
				throw new RuntimeException("file is not exists!");
			}
			
			fio = new FileInputStream(filePath);
			
			read = new InputStreamReader(fio, "UTF-8");// 考虑到编码格式

			BufferedReader br = null;
			
			List<String[]> strs = new ArrayList<String[]>();
			
			try {
				br = new BufferedReader(read);//构造一个BufferedReader类来读取文件
	            String s = null;
	            while((s = br.readLine())!=null){//使用readLine方法，一次读一行
	            	strs.add(s.split(breakSign));
	            }
			} catch (Exception e) {
				// TODO: handle exception
				e.printStackTrace();
			}finally {
				if(br!=null) {
					 br.close();    
				}
			}
			
            if(strs.size() <= 0) {
            	throw new RuntimeException("data size is 0.Please select the correct file!");
            }
            
            int dataSize = strs.size();
            
            int inputSize = channel * height * width;
            
            float[][] dataInput = new float[dataSize][inputSize];
            float[][] dataLabel = new float[dataSize][labelSize];
            String[] labels = new String[dataSize];
            
            for(int i = 0;i<strs.size();i++) {
            	String[] onceData = strs.get(i);
            	for(int j = 0;j<onceData.length;j++) {
            		if(j == onceData.length - 1) {
            			dataLabel[i][0] = Float.parseFloat(onceData[j]);
            			labels[i] = onceData[j];
            		}else {
            			dataInput[i][j] = Float.parseFloat(onceData[j]);
            		}
            	}
            	
            }
            
            return new DataSet(dataSize, channel, height, width, labelSize, dataInput, dataLabel, labels, null);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}finally {
			if(read!=null) {
				try {
					read.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			
			if(fio!=null) {
				try {
					fio.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			
		}
		
		return null;
	}
	
	/**
	 * loalDataByTxt(labelSet)
	 * @param filePath
	 * @param breakSign
	 * @param inputSize
	 * @param labelSize
	 * @param labelSet
	 * @return
	 */
	public static DataSet loalDataByTxt(String filePath,String breakSign,int channel,int height,int width,int labelSize,String[] labelSet) {
		
		InputStreamReader read = null;
		
		FileInputStream fio = null;
		
		try {
			
			if(filePath == null || filePath.equals("")) {
				throw new RuntimeException("filePath is null!");
			}
			
			if(labelSet==null || labelSet.length <= 0) {
            	throw new RuntimeException("labelSet size is 0!");
            }
			
			File file = new File(filePath);
			
			if(!file.exists()) {
				throw new RuntimeException("file is not exists!");
			}
			
			fio = new FileInputStream(filePath);
			
			read = new InputStreamReader(fio, "UTF-8");// 考虑到编码格式

			BufferedReader br = null;
			
			List<String[]> strs = new ArrayList<String[]>();
			
			try {
				br = new BufferedReader(read);//构造一个BufferedReader类来读取文件
	            String s = null;
	            while((s = br.readLine())!=null){//使用readLine方法，一次读一行
	            	strs.add(s.split(breakSign));
	            }
			} catch (Exception e) {
				// TODO: handle exception
				e.printStackTrace();
			}finally {
				if(br!=null) {
					 br.close();    
				}
			}
			
            if(strs.size() <= 0) {
            	throw new RuntimeException("data size is 0.Please select the correct file!");
            }
            
            int dataSize = strs.size();

            int inputSize = channel * height * width;
            
            float[][] dataInput = new float[dataSize][inputSize];
            float[][] dataLabel = new float[dataSize][labelSize];
            String[] labels = new String[dataSize];
            
            for(int i = 0;i<strs.size();i++) {
            	String[] onceData = strs.get(i);
            	for(int j = 0;j<onceData.length;j++) {
            		if(j == onceData.length - 1) {
            			dataLabel[i] = LabelUtils.labelToVector(onceData[j], labelSet);
            			labels[i] = onceData[j];
            		}else {
            			dataInput[i][j] = Float.parseFloat(onceData[j]);
            		}
            	}
            	
            }
            
            return new DataSet(dataSize, channel, height, width, labelSize, dataInput, dataLabel, labels, labelSet);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}finally {
			if(read!=null) {
				try {
					read.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			
			if(fio!=null) {
				try {
					fio.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			
		}
		
		return null;
	}
	
	/**
	 * getDataByUByte
	 * @param inputDataFileName
	 * @param labelDataFileName
	 * @return
	 */
	public static DataSet loadDataByUByte(String inputDataFileName,String labelDataFileName,String[] labelSet,int channel,int height,int width,boolean normalization) {
		
		try {
			
			float[][] dataInput = DataLoader.getImages(inputDataFileName,normalization);
			
			String[] labels = DataLoader.getLabels(labelDataFileName);
			
			if(dataInput!=null) {
				int dataSize = dataInput.length;
//				int inputSize = dataInput[0].length;
				int labelSize = labelSet.length;
				float[][] dataLabel = LabelUtils.labelToVector(labels, labelSet);
				return new DataSet(dataSize, channel, height, width, labelSize, dataInput, dataLabel, labels, labelSet);
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
		return null;
	}
	
	/**
	 * getDataByUByte
	 * @param inputDataFileName
	 * @param labelDataFileName
	 * @return
	 */
	public static DataSet loadDataByUByte(File inputDataFileName,File labelDataFileName,String[] labelSet,int channel,int height,int width,boolean normalization) {
		
		try {
			
			float[][] dataInput = DataLoader.getImages(inputDataFileName,normalization);
			
			String[] labels = DataLoader.getLabels(labelDataFileName);
			
			if(dataInput!=null) {
				int dataSize = dataInput.length;
//				int inputSize = dataInput[0].length;
				int labelSize = labelSet.length;
				float[][] dataLabel = LabelUtils.labelToVector(labels, labelSet);
				return new DataSet(dataSize, channel, height, width, labelSize, dataInput, dataLabel, labels, labelSet);
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
		return null;
	}
	
	/**
	 * 读取二进制文件
	 * @param inputDataFileName
	 * @param labelDataFileName
	 * @param labelSet
	 * @param channel
	 * @param height
	 * @param width
	 * @param normalization
	 * @return
	 */
	public static DataSet loadDataByBin(File inputDataFileName,File labelDataFileName,String[] labelSet,int channel,int height,int width,boolean normalization) {
		
		try {
			
			float[][] dataInput = DataLoader.getImages(inputDataFileName,normalization);
			
			String[] labels = DataLoader.getLabels(labelDataFileName);
			
			if(dataInput!=null) {
				int dataSize = dataInput.length;
//				int inputSize = dataInput[0].length;
				int labelSize = labelSet.length;
				float[][] dataLabel = LabelUtils.labelToVector(labels, labelSet);
				return new DataSet(dataSize, channel, height, width, labelSize, dataInput, dataLabel, labels, labelSet);
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
		return null;
	}
	
	/**
	 * getDataByUByte
	 * @param inputDataFileName
	 * @param labelDataFileName
	 * @return
	 */
	public static MatrixDataSet loadDataByUByteForMatrix(String inputDataFileName,String labelDataFileName,String[] labelSet,int channel,boolean normalization) {
		
		try {
			
			float[][][][] dataInput = DataLoader.getImagesToMatrix(inputDataFileName,channel,normalization);
			
			String[] labels = DataLoader.getLabels(labelDataFileName);
			
			if(dataInput!=null) {
				int dataSize = dataInput.length;
				int labelSize = labelSet.length;
				int height = dataInput[0][0].length;
				int width = dataInput[0][0][0].length;
				float[][] dataLabel = LabelUtils.labelToVector(labels, labelSet);
				return new MatrixDataSet(dataSize, channel, height, width, labelSize, dataInput, null, dataLabel, labels, labelSet);
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
		return null;
	}
	
	/**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static float[][] getImages(String fileName) {
        float[][] x = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000803".equals(bytesToHex(bytes))) {                        // 读取魔数
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);           // 读取样本总数
                bin.read(bytes, 0, 4);
                int xPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每行所含像素点数
                bin.read(bytes, 0, 4);
                int yPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每列所含像素点数
                x = new float[number][xPixel * yPixel];
                for (int i = 0; i < number; i++) {
                    float[] element = new float[xPixel * yPixel];
                    for (int j = 0; j < xPixel * yPixel; j++) {
//                        element[j] = bin.read();                                // 逐一读取像素值
                        // normalization
                        element[j] = bin.read() / 255.0f;
                    }
                    x[i] = element;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return x;
    }

	/**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static float[][] getImages(String fileName,boolean normalization) {
        float[][] x = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000803".equals(bytesToHex(bytes))) {                        // 读取魔数
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);           // 读取样本总数
                bin.read(bytes, 0, 4);
                int xPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每行所含像素点数
                bin.read(bytes, 0, 4);
                int yPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每列所含像素点数
                x = new float[number][xPixel * yPixel];
                for (int i = 0; i < number; i++) {
                    float[] element = new float[xPixel * yPixel];
                    for (int j = 0; j < xPixel * yPixel; j++) {                              // 逐一读取像素值
                        // normalization
                    	if(normalization){
                    		element[j] = bin.read() / 255.0f;
//                    		element[j] = (bin.read() - 127.5) / 128.0;
                    	}else {
                    		element[j] = bin.read();
                    	}
                    }
                    x[i] = element;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return x;
    }
    
    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static float[][] getImages(File file,boolean normalization) {
        float[][] x = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(file))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000803".equals(bytesToHex(bytes))) {                        // 读取魔数
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);           // 读取样本总数
                bin.read(bytes, 0, 4);
                int xPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每行所含像素点数
                bin.read(bytes, 0, 4);
                int yPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每列所含像素点数
                x = new float[number][xPixel * yPixel];
                for (int i = 0; i < number; i++) {
                    float[] element = new float[xPixel * yPixel];
                    for (int j = 0; j < xPixel * yPixel; j++) {                              // 逐一读取像素值
                        // normalization
                    	if(normalization){
                    		element[j] = bin.read() / 255.0f;
//                    		element[j] = (bin.read() - 127.5) / 128.0;
                    	}else {
                    		element[j] = bin.read();
                    	}
                    }
                    x[i] = element;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return x;
    }
    
    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static float[][][] getImagesToMatrix(String fileName,boolean normalization) {
        float[][][] x = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000803".equals(bytesToHex(bytes))) {                        // 读取魔数
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);           // 读取样本总数
                bin.read(bytes, 0, 4);
                int xPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每行所含像素点数
                bin.read(bytes, 0, 4);
                int yPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每列所含像素点数
                x = new float[number][yPixel][xPixel];
                for (int i = 0; i < number; i++) {
                    float[][] element = new float[yPixel][xPixel];
                    for (int py = 0; py < yPixel; py++) {
                    	for(int px = 0;px< xPixel;px++) {
                    		if(normalization) {
                    			element[py][px] = bin.read() / 255.0f;
                    		}else{
                    			element[py][px] = bin.read();
                    		}
                    	}
                    }
                    x[i] = element;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return x;
    }
    
    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static float[][][][] getImagesToMatrix(String fileName,int channel,boolean normalization) {
        float[][][][] x = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000803".equals(bytesToHex(bytes))) {                        // 读取魔数
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);           // 读取样本总数
                bin.read(bytes, 0, 4);
                int xPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每行所含像素点数
                bin.read(bytes, 0, 4);
                int yPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每列所含像素点数
                x = new float[number][channel][yPixel][xPixel];
                for (int i = 0; i < number; i++) {
                    float[][] element = new float[yPixel][xPixel];
                    for(int c = 0;c < channel; c++) {
                    	for (int py = 0;py<yPixel;py++) {
                        	for(int px = 0;px< xPixel;px++) {
                        		if(normalization) {
                        			element[py][px] = bin.read() / 255.0f;
                        		}else{
                        			element[py][px] = bin.read();
                        		}
                        	}
                        }
                        x[i][c] = element;
                    }
                    
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return x;
    }
    
    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static float[][][][] getImagesToMatrixByBin(String fileName,int number,int channel,int height,int width,boolean normalization) {
        float[][][][] x = new float[number][channel][height][width];
        
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {

        	for(int n = 0;n<number;n++) {
        		System.out.println(bin.read());
        		for(int c = 0;c<channel;c++) {
        			for(int h = 0;h<height;h++) {
        				for(int w = 0;w<width;w++) {
        					x[n][c][h][w] = bin.read();
        				}
        			}
        		}
        	}
        	
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        
        return x;
    }
    
    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static DataSet getImagesToDataSetByBin(String fileName,int number,int channel,int height,int width,int labelSize,boolean normalization,String[] labelSet) {
        float[][][][] x = new float[number][channel][height][width];
        String[] labels = new String[number];
        float[][] dataLabel = new float[number][labelSize];
        
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {

        	for(int n = 0;n<number;n++) {
        		int labelIndex = bin.read();
        		labels[n] = labelSet[labelIndex];
        		dataLabel[n] = LabelUtils.labelIndexToVector(labelIndex, labelSize);
        		for(int c = 0;c<channel;c++) {
        			for(int h = 0;h<height;h++) {
        				for(int w = 0;w<width;w++) {
        					if(normalization) {
//        						x[n][c][h][w] = (bin.read()&0xff)/128.0f-1;//normalize and centerlize(-1,1)
//        						x[n][c][h][w] = (float) (bin.read() / 255.0d) - 0.5f;
        						x[n][c][h][w] = (float) (bin.read() / 255.0d);
                    		}else{
                    			x[n][c][h][w] = bin.read();
                    		}
        				}
        			}
        		}
        	}
        	
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        
        DataSet data = new DataSet(number, channel, height, width, labelSize, x, dataLabel, labels, labelSet);
        
        return data;
    }
    
    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static DataSet getImagesToDataSetByBin(String fileName,int number,int channel,int height,int width,int labelSize,String[] labelSet,boolean normalization,float[] mean,float[] std) {
        float[][][][] x = new float[number][channel][height][width];
        String[] labels = new String[number];
        float[][] dataLabel = new float[number][labelSize];
        
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {

        	for(int n = 0;n<number;n++) {
        		int labelIndex = bin.read();
        		labels[n] = labelSet[labelIndex];
        		dataLabel[n] = LabelUtils.labelIndexToVector(labelIndex, labelSize);
        		for(int c = 0;c<channel;c++) {
        			for(int h = 0;h<height;h++) {
        				for(int w = 0;w<width;w++) {
        					if(normalization) {
//        						x[n][c][h][w] = (bin.read()&0xff)/128.0f-1;//normalize and centerlize(-1,1)
//        						x[n][c][h][w] = (float) (bin.read() / 255.0d) - 0.5f;
        						x[n][c][h][w] = (float) ((bin.read() / 255.0d) - mean[c]) / std[c];
                    		}else{
                    			x[n][c][h][w] = bin.read();
                    		}
        				}
        			}
        		}
        	}
        	
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        
        DataSet data = new DataSet(number, channel, height, width, labelSize, x, dataLabel, labels, labelSet);
        
        return data;
    }
    
    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static DataSet getImagesToDataSetByBin(String[] fileNames,int number,int channel,int height,int width,int labelSize,boolean normalization,String[] labelSet) {
        
    	int batchSize = number * fileNames.length;
    	
    	float[][][][] x = new float[batchSize][channel][height][width];
        String[] labels = new String[batchSize];
        float[][] dataLabel = new float[batchSize][labelSize];

    	int index = 0;
    	
        for(String fileName:fileNames) {

            try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            	
            	for(int n = 0;n<number;n++) {
            		int labelIndex = bin.read();
            		labels[index] = labelSet[labelIndex];
            		
            		dataLabel[index] = LabelUtils.labelIndexToVector(labelIndex, labelSize);
            		for(int c = 0;c<channel;c++) {
            			for(int h = 0;h<height;h++) {
            				for(int w = 0;w<width;w++) {
            					if(normalization) {
//            						x[index][c][h][w] = (bin.read()&0xff)/128.0f-1;//normalize and centerlize(-1,1)
//            						x[index][c][h][w] = (float) ((bin.read() / 255.0d) - 0.5);
            						x[index][c][h][w] = (float) (bin.read() / 255.0d) ;
                        		}else{
                        			x[index][c][h][w] = bin.read();
                        		}
            				}
            			}
            		}
            		index++;
            	}
            	
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            
        }
        
        DataSet data = new DataSet(batchSize, channel, height, width, labelSize, x, dataLabel, labels, labelSet);
        
        return data;
    }
    
    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static DataSet getImagesToDataSetByBin(String[] fileNames,int number,int channel,int height,int width,int labelSize,String[] labelSet,boolean normalization,float[] mean,float[] std) {
        
    	int batchSize = number * fileNames.length;
    	
    	float[][][][] x = new float[batchSize][channel][height][width];
        String[] labels = new String[batchSize];
        float[][] dataLabel = new float[batchSize][labelSize];

    	int index = 0;
    	
        for(String fileName:fileNames) {

            try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            	
            	for(int n = 0;n<number;n++) {
            		int labelIndex = bin.read();
            		labels[index] = labelSet[labelIndex];
            		
            		dataLabel[index] = LabelUtils.labelIndexToVector(labelIndex, labelSize);
            		for(int c = 0;c<channel;c++) {
            			for(int h = 0;h<height;h++) {
            				for(int w = 0;w<width;w++) {
            					if(normalization) {
//            						x[index][c][h][w] = (bin.read()&0xff)/128.0f-1;//normalize and centerlize(-1,1)
//            						x[index][c][h][w] = (float) ((bin.read() / 255.0d) - 0.5);
            						x[index][c][h][w] = (float) ((bin.read() / 255.0d) - mean[c]) / std[c] ;
                        		}else{
                        			x[index][c][h][w] = bin.read();
                        		}
            				}
            			}
            		}
            		index++;
            	}
            	
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            
        }
        
        DataSet data = new DataSet(batchSize, channel, height, width, labelSize, x, dataLabel, labels, labelSet);
        
        return data;
    }
    
    /**
     * get labels of `train` or `test`
     *
     * @param fileName the file of 'train' or 'test' about label
     * @return
     */
    public static float[] getLabelsTofloat(String fileName) {
        float[] y = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000801".equals(bytesToHex(bytes))) {
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);
                y = new float[number];
                for (int i = 0; i < number; i++) {
                    y[i] = bin.read();
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return y;
    }
    
    /**
     * get labels of `train` or `test`
     *
     * @param fileName the file of 'train' or 'test' about label
     * @return
     */
    public static String[] getLabels(String fileName) {
    	String[] y = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000801".equals(bytesToHex(bytes))) {
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);
                y = new String[number];
                for (int i = 0; i < number; i++) {
                    y[i] = bin.read() + "";
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return y;
    }
    
    /**
     * get labels of `train` or `test`
     *
     * @param fileName the file of 'train' or 'test' about label
     * @return
     */
    public static String[] getLabels(File file) {
    	String[] y = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(file))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000801".equals(bytesToHex(bytes))) {
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);
                y = new String[number];
                for (int i = 0; i < number; i++) {
                    y[i] = bin.read() + "";
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return y;
    }
    
    /**
     * change bytes into a hex string.
     *
     * @param bytes bytes
     * @return the returned hex string
     */
    public static String bytesToHex(byte[] bytes) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < bytes.length; i++) {
            String hex = Integer.toHexString(bytes[i] & 0xFF);
            if (hex.length() < 2) {
                sb.append(0);
            }
            sb.append(hex);
        }
        return sb.toString();
    }
	
    public static void main(String[] args) {
    	
    	String fileName = "E:/cifar-10-binary.tar/cifar-10-binary/cifar-10-batches-bin/test_batch.bin";

    	String[] labelSet = new String[] {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
    	
    	DataLoader.getImagesToDataSetByBin(fileName, 10000, 3, 32, 32, 10, false, labelSet);

    }
    
}
