# 零碎 python in DL 函数知识

## 2025/05/04

### 1. **nn.Embedding() vs nn.Linear()**

`nn.Embedding(num_embeddings, embedding_dim)` : 实现查表操作，把离散的整数索引映射为连续的稠密向量（嵌入向量）

- input : 整数索引

- 用于离散变量嵌入，稀疏更新

`nn.Linear(in_dim, out_dim)` : 实现一个线性变化，即 $WX + b$

- input : 连续张量


### 2. **.unsqueeze(dim) vs .squeeze(dim)**

`.unsqueeze(dim)` : 在指定的位置 `dim` 上增加一个维度（长度为 1）

- eg: 原张量 shape (4,5) : .unsqueeze(0) --> (1,4,5)  .unsqueeze(2) --> (4,5,1)

- pytorch 中，`tensor[:,None]` 是 `.unsqueeze()` 的一种快捷方式，用来在指定位置增加一个新的维度

- pytorch 张量索引中 `...` 即所有剩余维度

  - eg : x[0, :, :, :] == x[0,...] 等价 x--> (2,3,4,5)   

`squeeze(dim)` : 在指定位置上去掉长度为 1 的维度，若不为 1，不操作

`squeeze()` : 去掉所有维度 = 1 的维度

常见场景 ： 

- 批量输入单个值：有一个值 3.0，shape 是 ()`（标量）`，但 `nn.Linear(1, 64)` 需要 `(1, 1)`

- 模型输出单值预测 ： 模型输出为 `(batch_size, 1)`，但只想要 `(batch_size,)`

### 3. .long()

`.long()` : 把张量的 dtype 转成 `torch.int64`，也叫 LongTensor  （向下取整）

- 用途 : nn.embedding() : 输入的 “编号索引” 必须是整数，用来显式转换类型

### 4. 复制数据方法

`.expand(size)` : 不复制数据的情况下（共享内存），把一个小张量 “扩展” 成更大的张量视图。

- `.expand()` 常和 `.unsqueeze()` 一起用，因为 `.expand()` 只能在维度 = 1 的地方扩展

`.repeat()`

`.detach()` : 返回一个新的张量，和原来的数据共享内存（不会创建数据副本），但从计算图里 “分离” 出来，不再跟踪梯度


### 5. 数学计算

`torch.diff` : 沿指定维度计算 “相邻元素差值”

- 原型：`torch.diff(input, n=1, dim=1)` ：n : 差分阶数  dim : 进行差分的维度

- eg: x = torch.tensor([1, 2, 4, 7, 0]) 
    y = torch.diff(x)  ---> tensor([1, 2, 3, -7])

`torch.atan(input)` : 计算输入张量中每个元素的反正切值，不考虑象限，返回范围为 $(-\frac{\pi}{2}, \frac{\pi}{2})$

`torch.atan2(input, other)` : 计算输入张量中对应元素的二元反正切值，考虑象限，返回范围为 $(-\pi, \pi)$

`torch.clamp(input, min=None, max=None)` : 把张量中每个元素限制在指定的范围中，超过范围的部分会被 “夹紧”（clamp）到边界值

- 用途举例 ：
  
  - 限制概率范围，防止 log(0) 错误
 
  - 梯度裁剪以防止爆炸
  
`torch.max(input, dim)`：会沿指定的 `dim` 维度，找出最大值，并返回一个命名元组 (namedtuple)，包含两个部分: values 沿着这个维度的最大值， indices 最大值的索引位置 


### 6. Post-LN Transformer vs Pre-LN Transformer

**Post-LN** ：attention --> dropout --> residual --> layernorm 

- 特点 ： 经典（原论文中），稳定性稍弱（深层可能难训练）

**Pre-LN** : layernorm --> attention --> dropour --> residual

- 特点 ：梯度流更稳定，训练深层 Transformer 效果好

## 2025/05/05

### 7. nn.TransformerEncoder vs. nn.TransformerEncoderLayer

`nn.TransformerEncoderLayer` : 一个单独的 “编码器层”，即单层 Encoder Block，内部结构有 multi-head self-attention 子层和 feed forward 子层

- 结构流程 ： x --> [Multi-Head Attention] --> Dropout --> Add + LayerNorm --> [Feed Forward] --> Dropout --> Add + LayerNorm 

`nn.TransformerEncoder` : 一个完整的编码器模块，由多个 `TransformerEncoderLayer` 堆叠而成

- nn.TransformerEncoder(encoder_layer # 一层的定义, num_layers # 堆叠几层, norm=None # 最终的 LayerNorm)   复制 encoder_layer n 次，但权重彼此独立。

### 8. torch.nn.functional.unfold() : 用于将输入张量的滑动局部块 (patches) 提取出来，然后展平成二维张量形式

``` python
torch.nn.functional.unfold(
  input : Tensor 
  kernel_size,
  dilation=1,
  padding=0,
  stride=1
) -> Tensor  N * (kernel_size * kernel_size) * M
```

### 9. torch.cat vs torch.stack


## 2025/05/07 

### 10. torch.einsum(equation, *operands) 

`torch.einsum(equation, *operands)` : 基于爱因斯坦求和约定（Einstein summation convention）的通用张量运算接口，用一个简单的字符串公式同时表达多种张量操作

**Note**: 在处理大型张量运算时不如专门优化过的函数高效

**equation写法** : 指定张量操作的具体方式，由**输入标记**和**输出标记**组成，用 "->" 分隔，表示维度的字符只能是 26 个英文字母 'a' - 'z'

- 输入标记描述了输入张量的维度和形状，输出标记描述了输出张量的维度和形状，每个标记由一个或多个字母，用逗号分隔。equation 中字符也可以理解为索引，即输出张量的某个位置的值，是怎么从输入张量中得到的。

- 自由索引（free indices）和求和索引（summation indices）: 自由索引指出现在箭头右边的索引；求和索引指**只**出现在箭头左边的索引，表示中间计算结果需要在这个维度上求和之后才能得到输出。

- 求和准则：

  - rule1 : equation 箭头左边，在不同输入之间重复出现的索引，表示把输入张量沿着该维度做乘法操作
 
  - rule2 : 只出现在 equation 箭头左边的索引，表示中间计算结果需要在这个维度上求和，即求和索引
 
  - rule3 : equation 箭头右边索引顺序是任意的，

- 特殊规则 ：

  - special_rule1 : equation 可不写箭头内右边部分，输出张量的维度会根据默认规则推导，即把输入中只出现一次的索引取出来，然后按字母表顺序排列
 
  - special_rule2 : equation 中支持 "..." 省略号，表示不关心的索引 

Example : 用 torch.einsum() 实现卷积 

``` python
import torch
input = torch.arange(1,17).float().view(1, 1, 4, 4) # B * C * H * W
kernal = torch.tensor([[1.0, 0.0],[0.0, -1.0]]).view(1 , 1, 2, 2)

unfolded = torch.nn.functional.unfold(input, kernel_size=(2,2)) # 1 * 4 * 9 
output = torch.einsum('oc,bcn->bon', kernal.view(1,4), unfolded)

output_H = input.shape[2] - kernel.shape[2] + 1
output_W = input.shape[3] - kernel.shape[3] + 1
output = output.view(1, 1, output_H, output_W)

print(output)
``` 


更多使用示列见参考 blog。

> 参考 blog : https://blog.csdn.net/bj_zhb/article/details/136869289，吐槽下 gpt o4-mini 和 gemini advanced 2.5 pro，解释的快看到自闭 qwq

## 2025/05/08

### 11. torch.repeat_interleave() vs. tensor.repeat()

`torch.repeat_interleave(input:tensor, repeats: Union[int, Tensor], dim:int,)` : 对张量元素沿指定维度进行重复插值，将每个元素 “拉伸” 成多个副本，元素级别

`tensor.repeat(size)` : 整体进行复制，注意 $size_dim \geq tensor_dim$，若 size_dim > tensor_dim，默认在最左侧进行平铺  


## 2025/05/10

### 12. torch.multinomial() : 根据概率分布进行采样

``` python
torch.multinoimal(
  input,  # 输入的张量
  num_sample # 每个分布要抽取的样本数量
)
```
输入为矩阵时，把每一行理解为一个分布（不要求归一化），但要求非负

### 13. optimizer.zero_grad(set_to_none = False) set_to_none 参数

Pytorch 官方建议使用 `set_to_none = True` : 大多数情况下两者无明显差别，但使用 `set_to_none = True` 将梯度设置为 `None` 而不是一个零张量

- 稍微减少内存占用，带来微小性能提升

- 避免不必要操作（如优化器特定步骤或自定义操作等）以及更明确的梯度状态

### 14. print(loss.item()) 中 .item() 作用 ：

- 从一个只包含单个元素的 Pytorch 张量中提取除对应的标准 Python 数字

- 避免保留计算图，减少内存消耗

### 15. transformers.utils.ModelOutput

`ModelOutput` : 规范模型输出的格式，将输出组织成类似字典的数据类



## 2025/05/12 

### 16. weight tying（权重绑定）

**权重绑定** ：在神经网络中让不同部分的参数（权重）共享相同值的技术，当其中一个组件的这组共享参数在训练过程中更新时，其他绑定到这组参数的组件也会自动使用更新后的值。

**优点** :
- 减少模型参数量，从而降低模型的存储需求、减少过拟合的风险、减少梯度计算量来加快训练速度

- 提升模型的泛化能力，通过强制不同部分共享知识，模型可能学习到更通用、更鲁棒的表示。

- 引入归纳偏置，权重绑定向模型引入了一个强烈的先验假设，即共享权重的不同部分应该执行类似的功能或处理类似的信息。如果这个假设是合理的，那么权重绑定可以引导模型学到更好的解决方案。 

**著名应用**：

- 语言模型中：输入嵌入层 (Input Embedding Layer) 和输出层 (Output Layer / Logits Prediction Layer) 的权重绑定：

  - 背景：在语言模型中，输入层将词汇表中的单词（通常是one-hot编码或索引）映射到一个低维的密集向量表示（词嵌入）。输出层（通常在softmax激活函数之前）则需要将隐藏状态映射回词汇表大小的 logits，以预测下一个词的概率分布。
 
  - 如何绑定：输入嵌入矩阵的形状通常是 (词汇表大小, 嵌入维度)。输出层（在softmax之前）的线性变换层的权重矩阵的形状通常是 (隐藏层维度, 词汇表大小)。为了进行绑定，通常是将输入嵌入矩阵转置后用作输出层（或其一部分）的权重。如果嵌入维度和隐藏层维度不同，可能还需要一个额外的线性变换。
 
  - 直观理解：这种绑定的直观想法是，一个词作为输入时的表示（通过嵌入矩阵学习）和它作为输出被预测时的表示（通过输出层权重学习）应该是相关的，甚至可以是相同的。如果模型能很好地将一个词映射到其嵌入向量，那么反过来，它也应该能从一个相似的向量空间映射回那个词。 

### 17. adapter（适配器）

**适配器（aadapter）** : 一种参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT) 技术 

- 核心思想 : 在预训练模型中插入一些小型的、可训练的模块（即适配器），在微调时只更新这些适配器的参数，而保持预训练模型的主体参数冻结。

- 优点：大大减少需要训练的参数量，降低计算和存储成本

- 常见的适配器技术：LoRA, Prefix Tuning, AdapterHub 提出的瓶颈层适配器等

### 18. gradient checkpoint（梯度检查点）

**梯度检查点（gradient checkpoint）**：又称 activation checkpoint，一种在训练深度神经网络时用来减少显存占用的技术，核心思想是用计算换空间。

- 核心思想：在前向传播时，不存储所有的中间激活值，而只存储其中一部分（“检查点”）。在反向传播计算梯度时，如果遇到没有存储激活值的地方，就从最近的一个检查点开始，重新计算这部分前向传播，得到所需的激活值，然后再计算梯度。

### 19. early exit（早退出） & guard clause（卫语句）

**guard clause** ：

- 定义 ：卫语句是一种在函数或方法的开头，通过一系列条件检查来处理无效或边缘情况的编程模式。如果某个条件不满足（即输入无效或处于边缘情况），函数会立即返回或抛出异常，从而避免执行函数的主要逻辑。

- 目的 ：

  - 减少嵌套： 卫语句通过提前处理无效情况，避免了在函数主体中使用大量的 if-else if-else 嵌套结构。这使得代码的逻辑路径更加清晰
 
  - 提高可读性： 将前提条件检查放在函数开头，明确函数期望的输入和可能遇到的问题。
 
  - 关注核心逻辑： 将错误处理和边缘情况处理与核心逻辑分离开，使得核心逻辑更加突出和易于理解。

- 特点 ： 

  - 通常出现在函数的开头。
 
  - 条件检查通常是反向的（例如，检查 if (value == null) 而不是 if (value != null) 来继续执行）。
 
  - 如果条件满足（即发现无效情况），则立即退出函数
   

**early exit** ：

- 定义 ：一种更广泛的编程原则（卫语句是一种特定场景），指的是在函数或循环的执行过程中，一旦满足某个条件使得后续的计算或操作不再必要或不可能，就立即退出当前的执行块（函数、循环、switch 语句等）。


### 20. hasattr() & getattr()

`hasattr` 是 Python 内置函数，用来检查一个对象是否拥有指定名称的属性或方法。

``` python

hasattr(object, name) -> bool

object : 要检查的对象
name : 属性或方法的名字，必须是字符串类型

```

`getattr` 是 Python 内置函数，用来动态地从对象中获取属性或方法。

``` python

getattr(object, name[, default])  # [] 表示参数是可选的

object : 要获取属性的对象
name : 属性的名称，必须是字符串类型
default(optional) : 若指定的属性 name 在 object 中不存在，getattr() 会
      - 若提供 default 参数，则返回 default 值
      - 若未提供 default 参数，则引发 AttributeError 异常

```

## 2025/05/13 

### 21. torch.ne()

`torch.ne()` : 逐元素地比较两个张量，判断对应位置的元素是否**不相等**，然后返回对应的布尔张量，支持广播机制。

`torch.ne()` 与 `!=` 等价。

## 2025/05/19

### 22. field()

`field()` 是 Python 中 `dataclassed` 模块中的一个函数，用于自定义数据类中字段的行为，以实现更精细的控制。

主要用途 ： 

- 处理可变默认值，避免**可变对象**的所有实例共享同一对象 ：通过参数 `default_factory` 为可变对象提供一个无参数函数，每次创建类实例时调用该函数生成默认值

- 控制初始化行为 ：`default` 默认值，`init` `repr` 是否包含在 `__init__` `__repr__` 中

- 存储元数据 ：`metadata`

### 23. slow tokenizer vs. fast tokenizer

Hugging Face `transformers` 库通常提供两种类型的分词器实现 ：

- 基于 Python 的分词器（slow tokenizer）: 早期完全用 Python 实现的分词器，对于大规模数据集其处理速度较慢

- 快速分词器（Fast tokenizer）: 使用 Rust 重新实现的分词器，通过 `tokenizers` 库提供，追求更快的性能

### 24. streaming mode （流式模式）

**streaming mode** : 一种**按需处理**数据的机制，适用于处理非常大、无法一次性完全加载到内存中的数据集或数据流。通常为逐样本或逐批量加载。

### 25. bfloat16 vs. float16

**bfloat16** : 为神经网络专门设计的格式，牺牲精度换取更大的数值范围

**bfloat16** 与 **float16** 位宽分配区别

- float (IEEE 半精度)

  - 1 位符号位
 
  - 5 位指数位
 
  - 10 位尾数位 

- bfloat16 (Brain 浮点)

  - 1 位符号位
 
  - 8 位指数位
 
  - 7 位尾数位 

## 25/05/20

### 26. model sharding （模型切片） vs Data Parallel （数据并行）

**model sharding** :

- 将大型机器学习模型拆分为多个部分并分布在多个设备上进行训练和推理的技术。
 
- 实现方式 ：
  - tensor / parameter sharding ： 按参数张量切分

  - layer / module sharding (pipeline parallelism) ：按模型结构切分
 
  - hybrid sharding : 混合切分

**Data Parallel** :

- 每台机器/每块 GPU 有完整模型，各自计算一部分数据梯度然后全局同步

### 27. bitsandbytes、GPTQ、LLM.int8()

**bitsandbytes** : 由 Tim Dettmers 及其团队开发的一个高效的深度学习优化库，主要用于 PyTorch。核心功能为**支持高效的低比特量化**和**稀疏**技术。

**GPTQ(Generative Pre-trained Transformer Quantization)** : 针对大规模 Transformer 模型的高效权重量化，旨在实现极低比特宽度（如 4bit、3bit）下模型推理速度提升、显存减小且精度损失很小。

- 一般用于 **大模型的后训练**，不需要重新训练，只需对训练好的模型做量化。

**LLM.int8()** : 针对 LLM 的 8-bits 量化技术

- 传统 int8 量化直接将权重/激活全部 int8 量化，由于有些权重分布 “异常值” 对输出影响巨大，易造成精度大幅下降

- LLM.int8() 采用 mixed-precision dequantization（混合精度反量化），即针对 Transformer 的权重/激活分布，自动**区分重要和普通权重/激活**，普通采用 int8，重要数值保留位 float16/float32，推理时单独处理

### 28. llama.cpp、 vLLM

**llama.cpp** ：用 C/C++ 编写的，支持本地高效运行 Meta Llama 系列大型语言模型的开源推理引擎。纯 CPI 运行，极致轻量。

**vLLM** : 由加州大学伯克利分校开发的一个高效推理库，专为大语言模型在 GPU 上高性能推理设计，利用 "PagedAttention" 和 "动态 KV Cache"等创新技术极大提升 LLM 在生成文本时的吞吐量和并发能力，同时降低显存占用。

### 29. PEFT ：Parameter-Efficient Fine-Tuning

**PEFT** : 参数高效微调，由 HuggingFace 开发，专为大语言模型（LLM）、Transformer 等预训练模型设计的一套高效微调方法和工具包

### 30. .\*? （非贪婪匹配）  vs .* （贪婪匹配） 

`.` : 匹配任意单个字符（除了换行符）
`*` : 匹配前一个元素出现 0 次或多次
`?` : 匹配前一个元素出现 0 次或 1 次

`.*` : 匹配 “任意长度的任意字符序列” （贪婪模式，尽可能多） 

`.*?` ：任意字符，出现 0 次或多次，但尽可能少

> 补充 ：'\d+\.\d+'
>  `+` 表示出现 1 次或多次， '\.' 对 . 进行反转义 ，用来匹配一个小数

### 31. `::` 在 Python 切片

Python 中序列的切片一般格式如下 ：

''' python
sequence[start : stop : step]
- start : 起始下标（包括该位置，默认从 0 开始）
- stop : 结束下标（不包括该位置，默认到结尾）
- step : 步长（每隔多少个取一次，默认是 1）
'''

`:: step` 相当于 `start=None, stop=None, step=step`

### 32. adapter fusion 

**adapter fusion** : 一种用于**参数高效微调**技术

- 核心目标 ：利用多个后训练的 adapter，通过融合机制（如注意力、门控等）提升模型在多任务、多领域或少样本场景下的泛化与适应能力。

### 33. DeepSpeed 

**DeepSpeed** 是微软开发的一个开源**深度学习分布式训练与推理优化库**，主要用于提升大规模模型的训练和推理效率。支持多机多卡（分布式）、混合精度、模型并行、内存优化、超大批量训练等功能。

核心功能 ：

- ZeRO 优化器 (Zero Redundancy Optimizer)
  - 作用 ：将模型参数、优化器状态、梯度等拆分分布到多块 GPU 上（sharding），最大程度减少显存冗余，支持百亿/千亿参数模型在少量显卡上高效训练
  - 有三种 Stage :
    - Stage 1 : 参数状态 sharding
    - Stage 2 : 加上梯度 sharding
    - Stage 3 : 连模型权重本身也 sharding， 

- 大批量训练（Large Batch Training）
  - 支持极大批量（甚至数十万到百万级）的样本并行训练，通过梯度累积和优化显存管理提升吞吐量 













