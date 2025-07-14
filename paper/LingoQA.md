# LingoQA: Visual Question Answering for  Autonomous Driving

## What is the main contribution of this paper?

- **LingoQA Dataset**: 包含 419.9k QA pair, 问题和答案形式自由，覆盖感知和简单推理

- **LingoQA Benchmark**: 使用一个可学习的文本分类器用于评估，优于现存指标包括 GPT-4，与人类评估的 Spearman coefficient 为 0.950

> Spearman coefficient（斯皮尔曼相关系数）：用于衡量两个变量之间的单调关系（monotonic relationship），核心思想是不对原始数据进行计算，而是对数据的 “排名” 或 “等级” 及进行计算。

- **LingoQA Baseline**: 建立一个基线模型，

![LingoQA_example](./pictures/LingoQA_example.png)

## LingoQA Benchmark

**LingoQA benchmark** 由一个自动的评估度量和对应用来微调和评估的数据集组成。

### Evaluation Metric

现有评估方式严重依赖于 n-gram frequency 而忽略了潜在的答案的语义。

TODO


### Datasets

LingoQA dataset 包含 419.9k QA pairs，单个样本包含 4-second video clip at 1HZ。

![LingoQA_dataset_comparison](./pictures/LingoQA_dataset_comparison.png)

数据集包含两个部分：action dataset and scenery dataset。

![LingoQA_dataset_split](./pictures/Lingo_dataset_split.png)

**Action dataset** ：

- 根据记录的具有显著行为变化的驾驶场景创建的，场景注释由人工标注的高级描述和来自感知系统的原信息

- 场景注释由 GPT3.5 处理生成对当前动作及其理由的描述，以及为预期的答案生产示例问题和提示

- 通过基于动作和行为的策略对事件进行分类来重新平衡事件，并从每个类别中采样多达 500 个事件，产生 24577 个视频片段和 167，774 个 QA pairs

**Scenery dataset** :

- 场景数据集被设计用于补充动作数据集，聚焦于细粒度感知相关问题

- 采用 ELAN video annotation software 注释  30-minute driving sessions，对 sessions 提供大约 15 个不同类别的 brief captions

- 注释每秒每帧收集一次以构建文本描述，再使用 GPT-4 生产对感知问题的一系列思维链，43 QA pairs per video 

**Dataset statistics** :

- 训练数据集涵盖了 9 中不同的能力：action, justification, attention, identification, localisation, description,counting, anticipation, reasoning given contuerfactuals

![LingoQA_dataset_statistics](./pictures/LingoQA_dataset_statistics.png)

**Evaluation dataset**

- 收集一个人类反复标签、校正的数据集，以消除歧义

- 收集 1K high-quality answer 对应于 500 questions，每个问题有两个正确却多样的答案






















