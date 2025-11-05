package com.omega.boot.web.controller;

import com.omega.boot.starter.service.Yolov7Service;
import com.omega.boot.starter.utils.JarUrlUtils;
import com.omega.boot.starter.utils.YoloUtils;
import com.omega.boot.web.utils.FileUtils;
import org.slf4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Lazy;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

@RestController
@RequestMapping("/yolov7")
public class Yolov7Controller {

    private Logger logger = org.slf4j.LoggerFactory.getLogger(Yolov7Controller.class);

    @Lazy
    @Autowired
    private Yolov7Service yolov7Service;

    /**
     * 上传图片并调整尺寸
     * @param file 上传的文件
     * @return 上传结果
     */
    @PostMapping("/analyze")
    public ResponseEntity<Map<String, Object>> predict(
            @RequestParam("file") MultipartFile file) {

        Map<String, Object> response = new HashMap<>();

        try {
            // 检查文件是否为空
            if (file.isEmpty()) {
                response.put("success", false);
                response.put("message", "请选择要上传的文件");
                return ResponseEntity.badRequest().body(response);
            }

            // 检查文件类型
            String originalFilename = file.getOriginalFilename();
            if (!YoloUtils.yoloDetectImage(originalFilename) && !YoloUtils.yoloDetectVideo(originalFilename)) {
                response.put("success", false);
                response.put("message",  "仅支持图片："+String.join(",", YoloUtils.imageExtensions)+"格式，视频："+String.join(",", YoloUtils.vedioExtensions)+"格式");
                return ResponseEntity.badRequest().body(response);
            }

            // 创建上传目录
            Path uploadPath = Paths.get(JarUrlUtils.getJarPath() + File.separator + "upload" + File.separator + UUID.randomUUID().toString());
//            Path uploadPath = Paths.get("F://" + File.separator + "upload" + File.separator + UUID.randomUUID().toString());
            if (!Files.exists(uploadPath)) {
                Files.createDirectories(uploadPath);
            }

            // 生成唯一文件名
            String fileName = originalFilename;
            Path filePath = uploadPath.resolve(fileName);
            File targetFile = filePath.toFile();
            // 直接保存原图
            file.transferTo(targetFile);

            String result = yolov7Service.predict(targetFile.getAbsolutePath());
//            String result = targetFile.getAbsolutePath();
//            String result = "D:\\下载\\3000ecefa169d7a1624e1287d676eb22.jpg";

            // 构建响应
            response.put("success", true);
            response.put("message", "图片预测成功");
            response.put("fileName", fileName);
            response.put("fileUrl", FileUtils.getServerPath(result));

            return ResponseEntity.ok(response);

        } catch (IOException e) {
            logger.error("文件检测失败: {}", e);
            response.put("success", false);
            response.put("message", "文件上传失败: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        } catch (Exception e) {
            logger.error("文件检测失败: {}", e);
            response.put("success", false);
            response.put("message", "处理图片时发生错误: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }
}
