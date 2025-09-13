package com.omega.boot.web.controller;

import com.omega.boot.starter.service.Yolov3Service;
import com.omega.boot.starter.service.Yolov7Service;
import com.omega.boot.starter.utils.JarUrlUtils;
import com.omega.boot.web.utils.ImageUtils;
import com.omega.common.lib.LibPaths;
import org.apache.commons.io.FilenameUtils;
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
import java.security.CodeSource;
import java.security.ProtectionDomain;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/yolov7")
public class Yolov7Controller {

    @Lazy
    @Autowired
    private Yolov7Service yolov7Service;

    /**
     * 上传图片并调整尺寸
     * @param file 上传的文件
     * @return 上传结果
     */
    @PostMapping("/image")
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
            if (!ImageUtils.isImage(originalFilename)) {
                response.put("success", false);
                response.put("message", "只支持图片文件上传");
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

            String result = yolov7Service.predict(uploadPath.toFile().getAbsolutePath());
//            String result = targetFile.getAbsolutePath();
//            String result = "D:\\下载\\3000ecefa169d7a1624e1287d676eb22.jpg";

            byte[] fileContent = Files.readAllBytes(Path.of(result));

            // 转换为Base64
            String base64 = Base64.getEncoder().encodeToString(fileContent);

            // 构建响应
            response.put("success", true);
            response.put("message", "图片预测成功");
            response.put("fileName", fileName);
            response.put("fileSize", String.valueOf(fileContent.length));
            response.put("fileBase64", base64);

            return ResponseEntity.ok(response);

        } catch (IOException e) {
            response.put("success", false);
            response.put("message", "文件上传失败: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        } catch (Exception e) {
            response.put("success", false);
            response.put("message", "处理图片时发生错误: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }
}
