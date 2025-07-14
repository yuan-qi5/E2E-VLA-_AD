# GPT-Driver: Learning To Drive With GPT


## Induction

### What are major challenge that planner face ?

- process heterogeneous inputs, e.g. ego-vehicle information, maps, and perception results

- need to predict high-precision waypoint coordinates that represent a future driving

### How does this paper tackle these challenges ?

- 将 motion planning 看作 language modeling problem，将 heterogeneous planner inputs 对齐到 unified language tokens

- 提出 **prompting-reasoning-finetuning strategy**，让 GPT-3.5 先在自动驾驶场景中提示，然后进行思维链以产生合理输出，最后根据人类的驾驶轨迹对模型进行微调。能把误差控制在厘米级 


## GPT-Driver

![GPT_Driver_overview](./pictures/GPT_Driver_overview.png)


### Problem Definition

![learning-based_motion_formulated](./pictures/learning-based_motion_formulated.png)

虽然简单，但该方法试图同时回归不同尺度的航路点，例如，坐标值范围从 0 到 50 以上，通常回导致较远航路点的坐标估计不精确。

### Motion Planning As Language Modeling

展示如何重新制定运动规划作为一个语言建模问题，并缓解航路点坐标估计精度问题。

![reformulate_motion_planning](./pictures/reformulate_motion_planning.png)

从 tokenizer 角度可解释语言建模为何在运动规划中误差更小。以坐标 23.17 为例，通过 tokenizer，被分解为 "23", ".", "17"。因此预测这个航点坐标本质上是首先估计一个米级别的粗略位置（这里为 23），然后估计一个厘米级的细粒度位置（这里为 17）.此外这是通过对词汇表中正确的符号进行分类来建立估计，不是对它们的绝对值进行回归。

作者关键的观察是，一个常用的 language tokenizer，如 GPT tokenizer 已经有足够的能力去估计非常精确的数值运动规划，使得该方法更通用，也与自然语言兼容。

![GPT-Driver_example](./pictures/GPT-Driver_example.png)

### Prompting-Reasoning-Finetuning

介绍如何使用一种新的提示推理策略来解决语言建模问题


















