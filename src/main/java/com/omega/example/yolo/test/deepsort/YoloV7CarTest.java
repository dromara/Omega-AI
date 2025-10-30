package com.omega.example.yolo.test.deepsort;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.ComponentSampleModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.nio.ByteBuffer;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;

import com.omega.engine.loss.LossType;
import com.omega.engine.model.ModelLoader;
import com.omega.engine.nn.layer.YoloLayer;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.tensor.Tensor;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.yolo.data.ImageLoader;
import com.omega.example.yolo.model.YoloDetection;
import com.omega.example.yolo.utils.OMImage;
import com.omega.example.yolo.utils.YoloUtils;

/**
 * 车辆目标追踪示例
 */
public class YoloV7CarTest {

	private JPanel mainPanel = new JPanel(new BorderLayout());
	private SORTTracker tracker;
	private Yolo netWork;
	private Tensor input = new Tensor(1, 3, 416, 416, true);
	private final String[] labelset = new String[] { "car", "person", "bus", "others", "van" };

	private final String cfg_path = "D:\\workspace-ai\\omega-ai\\models\\yolov7-tiny-traffic.cfg";
	private final String model_path = "D:\\workspace-ai\\omega-ai\\models\\yolov7-traffic.model";
	private final String car_video_path = "E:\\car_detect.mp4";

	public static void main(String[] args) throws Exception {

		new YoloV7CarTest().run();
	}

	/**
	 * 
	 */
	public void run() throws Exception {

		JFrame mainFrame = new JFrame("Omega-ai-车辆追踪-演示");
		mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		mainFrame.add(mainPanel);
		JLabel loading = new JLabel("正在加载视频以及模型...");
		loading.setHorizontalAlignment(SwingConstants.CENTER);
		mainPanel.add(loading);
		mainFrame.setSize(500, 500);
		mainFrame.setLocationRelativeTo(null);
		mainFrame.setVisible(true);

		tracker = new SORTTracker();

		FFmpegFrameGrabber grabber = null;
		try {
			grabber = new FFmpegFrameGrabber(new File(car_video_path));
			grabber.setFrameRate(30);
			grabber.start();
		} catch (org.bytedeco.javacv.FFmpegFrameGrabber.Exception e) {
			e.printStackTrace();
		}

		Frame frame = null;

		Frame grabImage = grabber.grabImage();
		if (null == grabber || null == grabImage) {
			JLabel unsupport = new JLabel("不支持的解码格式");
			unsupport.setHorizontalAlignment(SwingConstants.CENTER);

			mainPanel.removeAll();
			mainPanel.add(unsupport);
			mainPanel.revalidate();
			mainPanel.repaint();

			return;
		}

		// 初始化
		BufferedImage firstFrame = toBufferedImage(grabImage);
		initYoloNet(firstFrame);

		while (true) {
			// 由于是本地视频，并且需要看视频，按照帧率处理
			frame = grabber.grabAtFrameRate();

			if (null == frame) {
				break;
			}

			if (frame.type != Frame.Type.VIDEO || null == frame.image) {
				frame.close();
				continue;
			}

			BufferedImage bufferedImage = toBufferedImage(frame);

			List<Detection> detections = runYOLODetection(bufferedImage);

			List<Track> activeTracks = tracker.update(detections);

			BufferedImage newBufferedImage = ImageTools.letterbox(bufferedImage, input.width, input.height);
			int sec = (int) (grabber.getTimestamp() / 1000f / 1000f);
			String formatTime = formatTime(sec);

			Set<Integer> set = new HashSet<>();
			for (Track track : activeTracks) {
				set.add(track.trackId);
			}

			for (Track track : activeTracks) {
				ImageTools.drawRect(newBufferedImage, track.bbox.x, track.bbox.y, track.bbox.width, track.bbox.height,
						Color.RED);
				ImageTools.drawText(newBufferedImage, track.trackId + "_" + track.label, track.bbox.x, track.bbox.y,
						Color.RED);
				ImageTools.drawText(newBufferedImage, "时间" + formatTime, 20, 20, Color.RED);
				//由于帧率设置为30，SORTTracker maxAge也为30，这里是初步估计的结果
				ImageTools.drawText(newBufferedImage, "每秒" + set.size() + "辆", 150, 20, Color.RED);
			}

			mainPanel.removeAll();
			mainPanel.add(new JLabel(new ImageIcon(newBufferedImage)));
			mainPanel.revalidate();
			mainPanel.repaint();

		}

		grabber.close();
	}

	/**
	 * 
	 * @param image
	 * @return
	 */
	private List<Detection> runYOLODetection(BufferedImage image) {

		List<Detection> detections = new ArrayList<>();

		OMImage orig = ImageLoader.loadImage(image);
		ImageLoader.loadVailDataDetection(input, null, 0, orig, null, input.width, input.height, 0, 0);
		input.hostToDevice();

		Tensor[] output = netWork.predicts(input);

		List<BoundingBox> list = new ArrayList<BoundingBox>();

		for (int i = 0; i < netWork.outputLayers.size(); i++) {
			YoloLayer layer = (YoloLayer) netWork.outputLayers.get(i);
			YoloDetection[][] dets = YoloUtils.getYoloDetectionsV7(output[i], layer.anchors, layer.mask, layer.bbox_num,
					layer.outputs, layer.class_number, netWork.getHeight(), netWork.getWidth(), 0.5f);
			for (int j = 0; j < dets.length; j++) {
				YoloDetection[] yoloDetections = dets[j];
				for (int k = 0; k < yoloDetections.length; k++) {
					YoloDetection yoloDetection = yoloDetections[k];
					if (yoloDetection.getObjectness() <= 0.7) {
						continue;
					}
					int classes = (int) yoloDetection.getClasses();
					BoundingBox boundingBox = new BoundingBox(yoloDetection.getBbox(), yoloDetection.getProb()[classes],
							classes);
					list.add(boundingBox);
				}
			}
		}

		List<BoundingBox> nms = BoundingBox.nms(list);

		for (BoundingBox boundingBox : nms) {

			int x = (int) boundingBox.getX(input.width);
			int y = (int) boundingBox.getY(input.height);
			int width = (int) boundingBox.getWidth(input.width);
			int height = (int) boundingBox.getHeight(input.height);

			Rect bbox = new Rect(x, y, width, height);

			detections.add(new Detection(bbox, boundingBox.confidence, labelset[boundingBox.predictedClass],
					boundingBox.predictedClass));
		}

		return detections;
	}

	/**
	 * 初始化yolo net
	 * 
	 * @param first
	 * @throws Exception
	 */
	public void initYoloNet(BufferedImage first) throws Exception {
		netWork = new Yolo(LossType.yolov7, UpdaterType.adamw);
		netWork.CUDNN = true;
		netWork.learnRate = 0.001f;
		ModelLoader.loadConfigToModel(netWork, cfg_path);
		netWork.init();
		ModelUtils.loadModel(netWork, model_path);

		OMImage orig = ImageLoader.loadImage(first);
		ImageLoader.loadVailDataDetection(input, null, 0, orig, null, input.width, input.height, 0, 0);
		input.hostToDevice();

		// 由于首次运行需要创建gpu空间，这边假设先拿一帧处理，进行第一次预测
		Tensor[] outputFirst = netWork.predicts(input);
	}

	public String formatTime(int seconds) {
        Duration duration = Duration.ofSeconds(seconds);
        long hours = duration.toHours();
        long minutes = duration.toMinutes();
        long remainingSeconds = duration.getSeconds() % 60;

        // 格式化输出为HH:mm:ss格式
//        String formattedTime = String.format("%02d:%02d:%02d", hours, minutes, remainingSeconds);
        return String.format("%02d:%02d", minutes, remainingSeconds);
	}
	
	public BufferedImage toBufferedImage(Frame frame) {
		ByteBuffer buffer = (ByteBuffer) frame.image[0].position(0);

		byte[] framePixels = new byte[buffer.limit()];
		buffer.get(framePixels);

		ColorSpace cs = ColorSpace.getInstance(ColorSpace.CS_sRGB);

		ColorModel cm = new ComponentColorModel(cs, false, false, Transparency.OPAQUE, DataBuffer.TYPE_BYTE);
		WritableRaster wr = Raster.createWritableRaster(new ComponentSampleModel(DataBuffer.TYPE_BYTE, frame.imageWidth,
				frame.imageHeight, frame.imageChannels, frame.imageStride, new int[] { 2, 1, 0 }), null);
		byte[] bufferPixels = ((DataBufferByte) wr.getDataBuffer()).getData();

		System.arraycopy(framePixels, 0, bufferPixels, 0, framePixels.length);

		return new BufferedImage(cm, wr, false, null);
	}

}
