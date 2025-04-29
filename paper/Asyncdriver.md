# Asynchronous Large Language Model Enhanced Planner for Autonomous Driving 


## Induction

currrent motion planning framework :

- learning-based real-time motion planning : 利用 vectorized map information 作为 input，employ a decoder 去预测轨迹

   - 易受长尾现象影响，在 rare or unseen scenarios 性能显著下降
 
   - 可控性低
     
- rule-based motion planning : 使用预先制定的规则去预测

  - 不足以捕捉整个潜在的复杂场景，易趋向于极端，要么过度谨慎，要么过度激进
 
  - 可控性低

- LLM-based motion planning : 使用 LLM 去预测，

  - 场景信息通过语言去描述，会受到 input token length 限制，使得捕捉精确且全面场景信息有挑战性
 
  - 语言输出预测轨迹不够精确 ：输出 high-level commands 容易引入误差，控制不够精确；直接用语言生成轨迹坐标，LLM 并不擅长精细的数值预测
 
  - prevalent framework 主要使用 LLMs 作为核心决策者，与 real-time planner 相比推理速度明显降低
    
![different_learning-based_AD_framework](./pictures/different_learning-based_AD_framework.png)

> nuPlan ：由 Motional 发布的大规模自动驾驶规划数据集，主要用于研究路径规划 (planning) 任务，特别强调在闭环训练和评估环境 (closed-loop training / evaluation) 中测试学习驱动的规划模型 (learning-based planner)
>
> learning-based planner : 指用机器学习方法（特别是深度学习）来学习自动驾驶的规划模块
>
> real-time planner : 能够在极短时间内（毫秒级）做出路径或轨迹决策的模块或算法
> 
> vectorized map（向量化地图）: 用几何向量形式（点、线、面）表示地图中的关键元素，而不是像素点的图像（raseter map）来表示地图信息，更像是用 “坐标 + 属性 + 拓扑” 构成的可计算对象
>
> raseter map（栅格地图）: 把地图以像素网格（pixels）形式存储的数据结构，本质是一个二维数据/图像矩阵，每个 pixel 包括 RGB 值（纯图像）和类别标签（语义地图）
>
> asynchronous（异步）：在编程中，异步表示多个任务并发执行而不是一个个排队顺序执行
>
> inference frequency（推理频率）：指 LLM 被调用进行推理的次数或频率
>
> model predictive control（模型预测控制）：一种基于模型的最优控制方法，核心思想：每一个控制时刻，利用系统动力学模型，在**未来一个时间窗口内预测系统行为**，通过**优化控制序列**使未来轨迹尽可能好（如偏离最小、控制能耗最小），只执行第一个控制动作，下一次再重新预测。
>   > intuition : MPC = 预测未来 + 求最优方案 + 实时滚动更新

## Data

nnPlan dataset : 第一个大规模自动驾驶规划基准，由 1200 hours 真实驾驶数据组成。

从 nnPlan Train and Val 中开发 pre-training and fine-tuning dataset ，集中于 14 个官方挑战性场景。

目的是为了增强 LLM 对自动驾驶中指令的理解能力。

### Pre-training Data Generation

- planning-QA
  
   - 由 rule-based 途径生成，为增强 LLM 对 waypoint、high-level instructions、control 间关系的理解。

   - 包含六种问题类型，每一种都侧重于在 waypoint、high-level instructions、control 的转换

- reasoning-1K

   - 由 GPT-4 生成的 1000 条数据，除了包含答案外还包含基于场景的推理与解释

### Fine-tuning Data Generation

- 为了进一步增强多模态理解和对齐，构建了一个基于 10,000 场景的微调数据集，每 8s 捕获一帧，得到了包含 180,000 帧的训练集和 20,000 帧的验证集，每一帧都包含 vectorized map data 和 linguistic prompts

   - 训练集和测试集中的场景类型分布与整个 nuPlan train-val dataset 分布保持一致 

-  对于 vectorized scene information，包含自车本身的信息，在 20 帧历史轨迹中 20 个周围智能体信息和椅子车为中心的全球地图数据

-  对于 LLM prompt ，由 system prompt 和 系列 routing instructions 组成。

   - 关于 routing instructions，使用基于规则的方法将路径点（pathway）转化为一系列带有距离信息的指令。训练集使用自车在未来 8 秒内的真实轨迹作为生成路径指令的基础，仿真时根据当前场景观察，通过人工设定方法，在规定的最大路径长度范围内，找到一条参考路径来生成路径指令。

## Methodology

AsyncDriver (asynchronous LLM-enhanced closed-loop framework): 由两部分组成 ： 

- Scene-Associated Instruction Feature Extraction Module

- Adaptive Injection Block

![asyncdriver_framework](./pictures/asyncdriver_framework.png)

### Scene-Associated Instruction Feature Extraction Module

**Multi-modal Input** : 

- 在每个 planning iteration，vectorized scene information 从模拟环境中获得。其中包括 ego 和 other agents 的历史轨迹和状态信息以及全球地图数据。

- real-time planner 的矢量化场景信息以相同方式提供，所有矢量数据都是相对于 ego 位置的。

- 随后通过 vector map encoder 和 map adapter 处理，map embeddings 和 language embeddings 一起送到 Llama2-13B 以得到 hidden features。

**Alignment Assistance Module**

### Adaptive Injection Block

 

### Asynchronous Inference 



### Training Details

## Experiment






