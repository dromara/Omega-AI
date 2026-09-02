# 在 SpringBoot 环境运行

Omega-AI 可以作为普通 Java 依赖放到 SpringBoot 项目中，用于模型加载、推理服务、离线任务或轻量训练。本篇提供集成思路，具体依赖坐标请以当前仓库构建产物为准。

[[toc]]

---

## 一、适用场景

| 场景 | 说明 |
| --- | --- |
| 推理服务 | 在接口中加载模型，对图片、文本或向量进行推理 |
| 离线任务 | 使用 SpringBoot 启动定时训练、批量推理、数据预处理 |
| 业务集成 | 将 Java 业务系统和 AI 能力放在同一技术栈中 |

## 二、添加依赖

如果项目已经发布到私服或本地 Maven 仓库，可以使用 Maven 引入：

```xml
<dependency>
    <groupId>io.gitee.iangellove</groupId>
    <artifactId>omega-engine-v4-gpu</artifactId>
    <version>1.0-beta</version>
</dependency>
```

如果还没有发布依赖，可以先在 Omega-AI 项目中执行：

```bash
mvn -DskipTests install
```

然后在 SpringBoot 项目中引用本地 Maven 仓库中的版本。

## 三、配置 GPU 动态库

SpringBoot 运行时同样需要找到 CUDA、cuDNN、JCuda 相关动态库。

Windows：

```properties
PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin;%PATH%
```

Linux：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
```

如果使用自定义 `.dll` 或 `.so`，建议放在应用启动时能被 JVM 找到的位置，或通过 `-Djava.library.path` 指定。

## 四、服务层示例

下面示例展示推荐的组织方式：模型初始化放在服务启动阶段，接口请求只做输入转换和推理调用。

```java
import org.springframework.stereotype.Service;

@Service
public class OmegaAiService {

    public OmegaAiService() {
        // 在这里初始化网络、加载权重、准备输入输出 Tensor。
        // 大模型不建议在每次请求中重复创建。
    }

    public float[] predict(float[] input) {
        // 1. 将业务输入转换为 Tensor。
        // 2. 调用 network.forward 或示例中对应的推理方法。
        // 3. 将输出 Tensor 转成业务结果。
        return new float[0];
    }
}
```

## 五、接口示例

```java
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class OmegaAiController {

    private final OmegaAiService omegaAiService;

    public OmegaAiController(OmegaAiService omegaAiService) {
        this.omegaAiService = omegaAiService;
    }

    @PostMapping("/predict")
    public float[] predict(@RequestBody float[] input) {
        return omegaAiService.predict(input);
    }
}
```

## 六、集成建议

- 模型和权重只初始化一次，不要在接口中反复加载。
- GPU 推理服务建议固定 batch 或做简单队列合批。
- 图片、文本预处理尽量在 CPU 侧完成，进入模型前统一归一化。
- 大模型训练任务不建议直接挂在 HTTP 请求里，推荐做成后台任务。
- 线上服务需要明确显存上限、并发上限和异常恢复策略。
