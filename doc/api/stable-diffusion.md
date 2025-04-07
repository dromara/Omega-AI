# Stable Diffusion 稳定扩散模型

## 概述
Stable Diffusion通过潜在扩散过程实现文本到图像生成，核心创新包括：
- 潜在空间扩散降低计算成本
- CLIP文本编码器实现文本引导
- 安全过滤器防止不当内容生成

## 核心结构
```java
/**
 * 稳定扩散基础实现
 */
public class StableDiffusion {
    private AutoencoderKL vae;          // 图像编解码器
    private CLIPTextModel textEncoder;  // 文本编码器
    private UNet2DConditionModel unet;  // 条件扩散模型
    private Scheduler scheduler;        // 噪声调度器
    
    // 生成参数
    private int numInferenceSteps = 50;
    private float guidanceScale = 7.5f;
}
```

## 完整实现

### 1. 潜在扩散过程
```java
public class LatentDiffusion {
    public float[][] diffuse(float[][] latents, String prompt) {
        // 文本编码
        float[][] textEmbeds = textEncoder.encode(prompt);
        
        // 扩散循环
        for (int t=0; t<numInferenceSteps; t++) {
            // 合并无条件和条件预测
            float[][] noisePred = applyClassifierFreeGuidance(
                latents, t, textEmbeds);
            
            // 调度器更新
            latents = scheduler.step(noisePred, t, latents);
        }
        return vae.decode(latents);
    }
    
    private float[][] applyClassifierFreeGuidance(float[][] latents, int t, 
                                                 float[][] textEmbeds) {
        // 无条件预测
        float[][] uncond = unet(latents, t, null);
        // 条件预测
        float[][] cond = unet(latents, t, textEmbeds);
        return uncond + guidanceScale * (cond - uncond);
    }
}
```

### 2. 安全过滤
```java
public class SafetyChecker {
    public boolean isSafe(float[][] image) {
        // NSFW内容检测
        float nsfwScore = calculateNSFWScore(image);
        // 血腥内容检测
        float violenceScore = calculateViolenceScore(image);
        return nsfwScore < 0.5 && violenceScore < 0.7;
    }
}
```

## 使用示例（文生图）
```java
public class TextToImage {
    public static void main(String[] args) {
        // 初始化模型
        StableDiffusion model = StableDiffusion.load("sd-v1.5");
        
        // 生成参数配置
        model.setSeed(42)
            .setSteps(30)
            .setSize(512, 512);
        
        // 生成图像
        String prompt = "赛博朋克风格的城市夜景，霓虹灯闪烁，下雨的街道";
        float[][] image = model.generate(prompt);
        
        // 安全过滤
        if (model.isSafe(image)) {
            saveAsJPG(image, "output.jpg");
        } else {
            System.out.println("内容不安全，已过滤");
        }
    }
}
```

## 性能优化
1. **内存优化**：分块推理
```java
public class MemoryOptimizer {
    public float[][] generateLargeImage(String prompt) {
        // 将生成过程分割为4个区块
        return mergeBlocks(
            generateBlock(prompt, 0, 0),
            generateBlock(prompt, 256, 0),
            generateBlock(prompt, 0, 256),
            generateBlock(prompt, 256, 256)
        );
    }
}
```

2. **模型加速**：ONNX导出
```java
public void exportToONNX() {
    this.unet.export("unet.onnx");
    this.vae.export("vae.onnx");
    this.textEncoder.export("clip.onnx");
}
```

## 常见问题
### Q1：生成图像模糊如何改善？
- 高分辨率修复方案：
```java
public float[][] hiresFix(float[][] image) {
    float[][] upscaled = upscale(image, 2.0f);
    return refine(upscaled, 10); // 额外细化10步
}
```

### Q2：长文本提示效果差？
- 文本聚焦实现：
```java
public void emphasizeKeywords(String prompt) {
    // 增强关键词："(关键词:1.5)"
    String processed = prompt.replaceAll("关键词", "(关键词:1.5)");
    return model.generate(processed);
}
```

## 扩展阅读
- [ControlNet控制网络](/doc/extension/controlnet)
- [LoRA低秩适配器](/doc/extension/lora)
- [模型蒸馏加速](/doc/extension/distillation)