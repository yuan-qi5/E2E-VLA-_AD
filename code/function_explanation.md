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




