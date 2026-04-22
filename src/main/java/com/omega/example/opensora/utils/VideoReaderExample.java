package com.omega.example.opensora.utils;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import com.omega.common.utils.ImageUtils;
import com.omega.engine.tensor.Tensor;

/**
 * 视频读取和写入示例
 */
public class VideoReaderExample {
	
	public static Tensor loadVideo2Tesnro(String videoPath, int maxFrames, int targetHeight, int targetWidth) {
		
		VideoReader reader = new VideoReader(videoPath, 10);
		
		float[] mean = new float[] {0.5f, 0.5f, 0.5f};
		float[] std = new float[] {0.5f, 0.5f, 0.5f};
		
		try {
			 // 1. 打开视频
            reader.open();

            // 2. 获取视频信息
            VideoReader.VideoInfo info = reader.getVideoInfo();
            System.out.println("\n" + info);
            System.out.println();

            // 3. 使用新方法顺序读取所有帧（按30FPS采样）
            System.out.println("开始读取帧（按30FPS采样）...");

            List<BufferedImage> frames = reader.readAllFramesByFPS(maxFrames);
            
            // 方式2: 自定义宽高输出 (例如 320x240)
            System.out.println("\n开始 Resize (CenterCrop " + targetWidth + "x" + targetHeight + ")...");
            List<BufferedImage> resizedFrames = VideoReader.resizeCenterCrop(frames, targetWidth, targetHeight);
            
            Tensor x = new Tensor(1, 3 * (int) maxFrames, targetHeight, targetWidth, true);
            
            List<float[]> frameRGB = new ArrayList<float[]>();
            
            for(int f = 0;f<maxFrames;f++) {
            	int rf = f;
            	if(f >= resizedFrames.size()) {
            		rf = resizedFrames.size() - 1;
            	}
            	float[] rgb = ImageUtils.getImageData(resizedFrames.get(rf), true, true, mean, std);
            	frameRGB.add(rgb);
            }
            int os = maxFrames * targetHeight * targetWidth;
            for(int c = 0;c<3;c++) {
            	for(int f = 0;f<maxFrames;f++) {
            		float[] rgb = frameRGB.get(f);
            		for(int s = 0;s<targetHeight*targetWidth;s++) {
            			x.data[c * os + f * targetHeight * targetWidth + s] = rgb[c * targetHeight * targetWidth + s];
            		}
            	}
            }
            
            x.hostToDevice();
            return x;
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		return null;
	}
	
	public static void loadVideo2Tensor(String videoPath, int maxFrames, int targetHeight, int targetWidth, Tensor input, int index) {
		
		VideoReader reader = new VideoReader(videoPath, 10);
		
		float[] mean = new float[] {0.5f, 0.5f, 0.5f};
		float[] std = new float[] {0.5f, 0.5f, 0.5f};
		
		try {
			 // 1. 打开视频
            reader.open();

            // 2. 获取视频信息
//            VideoReader.VideoInfo info = reader.getVideoInfo();
//            System.out.println("\n" + info);
//            System.out.println();

            // 3. 使用新方法顺序读取所有帧（按30FPS采样）
//            System.out.println("开始读取帧（按30FPS采样）...");

            List<BufferedImage> frames = reader.readAllFramesByFPS(maxFrames);
            
            // 方式2: 自定义宽高输出 (例如 320x240)
//            System.out.println("\n开始 Resize (CenterCrop " + targetWidth + "x" + targetHeight + ")...");
            List<BufferedImage> resizedFrames = VideoReader.resizeCenterCrop(frames, targetWidth, targetHeight);
            
            List<float[]> frameRGB = new ArrayList<float[]>();
            
            for(int f = 0;f<maxFrames;f++) {
            	int real_f = f;
            	if(f >= resizedFrames.size()) {
            		real_f = resizedFrames.size() - 1;
            	}
            	float[] rgb = ImageUtils.getImageData(resizedFrames.get(real_f), true, true, mean, std);
            	frameRGB.add(rgb);
            }
            int os = maxFrames * targetHeight * targetWidth;
            for(int c = 0;c<3;c++) {
            	for(int f = 0;f<maxFrames;f++) {
            		float[] rgb = frameRGB.get(f);
            		for(int s = 0;s<targetHeight*targetWidth;s++) {
            			input.data[index * 3 * os + c * os + f * targetHeight * targetWidth + s] = rgb[c * targetHeight * targetWidth + s];
            		}
            	}
            }

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
    public static void main(String[] args) {
        // 视频文件路径 - 请替换为实际的视频路径
        String videoPath = "D:\\dataset\\wfvae\\4105473_scene-0_cut-border.mp4";
        String outputPath = "D:\\video_project\\output\\result_video.mp4";

        // 创建视频读取器，帧率设为30FPS
        VideoReader reader = new VideoReader(videoPath, 30);

        try {
            // 1. 打开视频
            reader.open();

            // 2. 获取视频信息
            VideoReader.VideoInfo info = reader.getVideoInfo();
            System.out.println("\n" + info);
            System.out.println();

            // 3. 使用新方法顺序读取所有帧（按30FPS采样）
            System.out.println("开始读取帧（按30FPS采样）...");
            long maxFrames = 17; // 限制读取帧数
            List<BufferedImage> frames = reader.readAllFramesByFPS((int) maxFrames);

            System.out.printf("\n总共读取了 %d 帧%n", frames.size());

            // 4. Resize 处理 - UCFCenterCropVideo 方式
            // 方式1: 正方形输出 256x256
//            int targetSize = 256;  // 常用的 UCF101 数据集尺寸
//            System.out.println("\n开始 Resize (CenterCrop " + targetSize + "x" + targetSize + ")...");
//            List<BufferedImage> resizedFrames = VideoReader.resizeCenterCrop(frames, targetSize);

            // 方式2: 自定义宽高输出 (例如 320x240)
             int targetWidth = 640;
             int targetHeight = 352;
             System.out.println("\n开始 Resize (CenterCrop " + targetWidth + "x" + targetHeight + ")...");
             List<BufferedImage> resizedFrames = VideoReader.resizeCenterCrop(frames, targetWidth, targetHeight);

            // 保存原始帧和resize后的帧为图片（可选）
            for (int i = 0; i < resizedFrames.size(); i++) {
                BufferedImage frame = frames.get(i);
                BufferedImage resizedFrame = resizedFrames.get(i);
                System.out.printf("帧 #%d: 原始尺寸 %dx%d -> Resize后 %dx%d%n",
                        i + 1, frame.getWidth(), frame.getHeight(),
                        resizedFrame.getWidth(), resizedFrame.getHeight());

                saveFrame(frame, "D:\\video_project\\output\\frame_original_" + (i + 1) + ".jpg");
                saveFrame(resizedFrame, "D:\\video_project\\output\\frame_resized_" + (i + 1) + ".jpg");
            }

            // 5. 将 Resize 后的 BufferedImage 数组转回视频
            System.out.println("\n开始写入视频...");
            writeFramesToVideo(resizedFrames, outputPath);

        } catch (Exception e) {
            System.err.println("处理视频时出错: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // 6. 关闭视频
            try {
                reader.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * 将 BufferedImage 数组写入视频文件
     */
    public static void writeFramesToVideo(List<BufferedImage> frames, String outputPath) throws Exception {
        // 根据第一帧的尺寸自动获取宽高
        int width = frames.get(0).getWidth();
        int height = frames.get(0).getHeight();

        System.out.println("输出视频尺寸: " + width + "x" + height);

        VideoWriter writer = new VideoWriter(outputPath, 10, width, height);

        // 可选：设置编码质量和速度
        // writer.setQuality(23);  // CRF值 (1-51, 默认23)
        // writer.setPreset("medium");  // 编码速度预设

        // 写入所有帧
        writer.writeAll(frames);

        // 释放资源
        writer.release();
    }

    /**
     * 保存帧为图片文件
     */
    private static void saveFrame(BufferedImage frame, String outputPath) throws IOException {
        File outputFile = new File(outputPath);
        outputFile.getParentFile().mkdirs();
        ImageIO.write(frame, "jpg", outputFile);
    }
}
