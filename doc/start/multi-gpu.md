# 多卡训练配置

Omega-AI 支持在多 GPU 环境中进行训练探索。多卡训练涉及数据切分、模型参数同步、梯度同步和显存管理，建议先跑通单卡训练，再切换到多卡。

[[toc]]

---

## 一、训练前检查

```bash
nvidia-smi
```

确认输出中能看到多张 GPU，并记录每张卡的显存、驱动版本和当前占用。

## 二、常见启动方式

如果示例代码内部支持指定 GPU，可以在 Java 参数或配置文件中传入设备编号。不同示例的参数名称可能不同，请以对应 `main` 方法为准。

Linux 常用方式：

```bash
CUDA_VISIBLE_DEVICES=0,1 java -Xms16g -Xmx16g -cp target/xxx.jar com.omega.example.xxx.TrainDemo
```

Windows PowerShell：

```powershell
$env:CUDA_VISIBLE_DEVICES="0,1"
java -Xms16g -Xmx16g -cp target\xxx.jar com.omega.example.xxx.TrainDemo
```

## 三、需要关注的参数

| 参数 | 建议 |
| --- | --- |
| batch size | 先按单卡可承受 batch 估算，再乘以卡数 |
| learning rate | 总 batch 变大时需要重新验证学习率 |
| 数据加载线程 | 多卡训练更容易被数据加载拖慢 |
| 梯度同步 | 需要确认同步发生在 optimizer 更新前 |
| 随机种子 | 对比单卡和多卡时建议固定 |

## 四、排查思路

多卡训练如果没有提速，优先检查：

1. 数据加载是否成为瓶颈。
2. 每张 GPU 的利用率是否均衡。
3. 梯度同步是否过于频繁。
4. batch size 是否太小，导致通信开销掩盖计算收益。
5. 是否存在 CPU/GPU 同步点，例如频繁 `syncHost()`。

## 五、建议

- 先让单卡训练稳定，再扩展多卡。
- 单步耗时建议用 nsys 或日志拆分成数据加载、forward、backward、update。
- 多卡效率低时，优先增大每卡 batch 或减少不必要的 CPU 同步。
