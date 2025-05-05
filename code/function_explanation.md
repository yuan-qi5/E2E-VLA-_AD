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

### 8. torch.cat vs torch.stack



