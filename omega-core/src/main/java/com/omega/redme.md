
```
├── omega-common/       # 基础公用模块
│
├── omega-core/         # 核心模块
│
├── omega-cuda          # cuda 相关
│
├── omega-data/         # 暂定
│
├── omega-examples/     # 项目使用案例          
│
├── omega-utils         # （工具）暂定
│

```
### omega-core模块包结构
```

├── common/                   # 框架基层 公共底层代码（常量，张量，网络层等基础类型定义 ，张量操作运算实现方法（加减乘除，cpuToGpu..），）(后期抽成单独基础的模块)
│  ├──config                  # 配置
│  ├──lib                     # 
│  ├──task                    # 
│  ├──tensor                  # 张量
│  ├──utils                   # 基础工具类
├── models/                   # 模型相关
│
├── data/                     # 数据集相关
│
├── scripts/                  # 训练和评估相关的脚本
│
├── utils/                    # 工具函数
│
├── cuda/                     # cuda相关代码
│
├── configs/                  # 配置文件
│
├── experiments/test          # 测试实验
│
├── resources/                # 资源文件

大家可以讨论 针对 神经网络这块 在创建相应的包 比如nn 激活函数，激动求导，针对asr  那种音频格式处理，是否单独分包
大家对于分包有更好的建议 再群里提出，或者修改后群里通知一下大家更新代码，避免冲突。
src 的源代码暂时保留 防止我拆包导致代码错误，后续删除。
```

