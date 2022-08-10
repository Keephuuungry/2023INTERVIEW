# Transformer Q & A

[TOC]



> https://zhuanlan.zhihu.com/p/438625445
>
> https://zhuanlan.zhihu.com/p/363466672
>
> https://zhuanlan.zhihu.com/p/148656446
>
> https://github.com/DA-southampton/NLP_ability
>
> https://github.com/DA-southampton/NLP_ability/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/Transformer/%E7%AD%94%E6%A1%88%E8%A7%A3%E6%9E%90(1)%E2%80%94%E5%8F%B2%E4%B8%8A%E6%9C%80%E5%85%A8Transformer%E9%9D%A2%E8%AF%95%E9%A2%98%EF%BC%9A%E7%81%B5%E9%AD%8220%E9%97%AE%E5%B8%AE%E4%BD%A0%E5%BD%BB%E5%BA%95%E6%90%9E%E5%AE%9ATransformer.md
>
> https://zhuanlan.zhihu.com/p/69697467 代码细节
>
> http://nlp.seas.harvard.edu/2018/04/03/attention.html代码
>
> https://github.com/NLP-LOVE/ML-NLP/tree/master/NLP/16.7%20Transformer

### Q1：为什么使用不同的`K`和`Q`，而不使用同一个值（`K`自己点乘？）

通俗来说，问题想表达的是：既然K和Q差不多（唯一区别是$W^K$、$W^Q$权值不同），直接拿K自己点乘就行，何必要再创建一个Q？
$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{(d_k)}})V
$$
想要回答这个问题，我们首先要明白，为什么要计算Q和K的点乘。

首先补充两点：

1. 先从点乘的物理意义说，两个向量的点乘表示两个向量的相似度。
2. Q、K、V物理意义上是一样的，都表示同一个句子不同token组成的矩阵。矩阵的每一行是表示一个token的word embedding向量。假设一个句子“Hello，how are you？”长度是6，embedding维度为300，假设Q、K、V长度均为64，则与大小为(300, 64)的$W^Q$、$W^K$、$W^V$相乘后，得到的Q、K、V向量都是（6,64）的矩阵。

简单的说，K和Q的点乘是为了计算一个句子中每个token相对于句子中其他token的相似度，这个相似度可以理解为attention score，关注度得分。比如说 "Hello, how are you?"这句话，当前token为”Hello"的时候，我们可以知道”Hello“对于” , “, "how", "are", "you", "?"这几个token对应的关注度是多少。有了这个attetnion score，可以知道处理到”Hello“的时候，模型在关注句子中的哪些token。而后添加的softmax只是为了对关注度进行归一化。

经过上面的解释，我们知道K和Q的点乘是为了得到一个attention score矩阵，用来对V进行提纯。K和Q使用了不同的$W^K$、$W^Q$来计算，可以理解为是在不同空间上的投影。正因为有了这种不同空间的投影，增加了表达能力，这样计算得到的attention score矩阵的泛化能力更高。

如果不用Q，直接拿K和K点乘的话，会发现attention score矩阵是一个对称矩阵。

- 对称带来的束缚就是A对B的重要程度和B对A的重要程度是相同的。这往往不对。比如说"I went to office yesterday"。 "went"对yesterday可能不是很重要, 但是"yesterday"对"went"很重要；如果K和Q使用不同的值的话, A对B的重要程度是Key(A) * Query(B), 而B对A的重要程度是Key(B) * Query(A). 可以看出, A对B的重要程度与B对A的重要程度是不同的。

  > 很多答主都提到了A对B的重要程度和B对A的重要程度不同的问题。但是忽略了重要值和重要程度的区别。A对B的重要程度如果要和B对A的重要程度一样，并不是由dot_product(query_A,key_B) ==? dot_product(query_B,key_A)决定的，因为重要程度在这里的定义肯定不是一个重要值，而是row-wise的softmax后重要值占比。
  >
  > 链接：https://www.zhihu.com/question/319339652/answer/2192385398

- 对称矩阵这一约束一定程度上缩小了这个矩阵的变化空间，也可以理解为对一个无约束的矩阵加了个正则。

### Q2：Self-Attention和cross-Attention的区别与联系

### Q3：为什么在进行softmax之前需要对attention进行scaled

> 参考：https://icode.best/i/45091844575359
>
> https://www.zhihu.com/question/339723385

#### 为什么比较大的输入会使得softmax的梯度变得很小？

对于一个输入向量$x$，softmax将其映射/归一化到一个分布$\widehat{y}$。在这个过程中，softmax先用一个自然底数e将输入中的元素间差距先拉大，然后归一化为一个分布。假设某个输入x中最大的元素下标是k，如果输入的数量级变大（每个元素都很大），那么$\widehat{y}_k$会非常接近1。

我们可以用一个小例子来看看x的数量级对输入最大元素对应的预测概率$\widehat{y}_k$的影响。假定输入$x=[a,a,2a]^T$，我们来看看不同量级的a产生的$\widehat{y}_3$有什么区别：

- a=1时，$\widehat{y}_3=0.5761168847658291$;
- a=10时，$\widehat{y}_3=0.999909208384341$；
- a=100时，$\widehat{y}_3\approx1.0$。

<img src="Transformer Q & A.assets/c227a8951141441c9df6175e47cf20c3.png" alt="在这里插入图片描述" style="zoom: 33%;" />

可以看到，数量级对softmax得到的分布影响非常大。在数量级较大时，softmax将几乎全部的概率分布都分配给了最大值对应的标签。记softmax函数为$g(\cdot)$.

$\widehat{y}=g(x)$对输入x的梯度为：
$$
\frac{\partial g(x)}{\partial x}=diag(\widehat{y})-\widehat{y}\widehat{y}^T\\
a = \left[
\matrix{
  \widehat{y}_1 & 0 & \cdot\cdot\cdot & 0\\
  0 & \widehat{y}_2 & \cdot\cdot\cdot & 0\\
  \cdot & \cdot & \cdot\cdot\cdot & \cdot\\
  0 & 0 & \cdot\cdot\cdot & \widehat{y}_d
}
\right]-
\left[
\matrix{
  \widehat{y}_1^2 & \widehat{y}_1\widehat{y}_2 & \cdot\cdot\cdot & \widehat{y}_1\widehat{y}_d\\
  \widehat{y}_2\widehat{y}_1 & \widehat{y}_2^2 & \cdot\cdot\cdot & \widehat{y}_2\widehat{y}_d\\
  \cdot & \cdot & \cdot\cdot\cdot & \cdot\\
  \widehat{y}_d\widehat{y}_1 & \widehat{y}_d\widehat{y}_2 & \cdot\cdot\cdot & \widehat{y}_d^2
}
\right]
$$
根据前面讨论，当输入x的元素均较大时，softmax会把大部分概率分布分配给最大的元素，假设我们的输入数量级很大，最大的元素是$x_1$，那么就将产生一个接近one-hot向量的$\widehat{y}\approx[1,0,...,0]^T$，此时上面的矩阵变为如下形式：
$$
\frac{\partial g(x)}{\partial x}\approx \left[
\matrix{
  1 & 0 & \cdot\cdot\cdot & 0\\
  0 & 1 & \cdot\cdot\cdot & 0\\
  \cdot & \cdot & \cdot\cdot\cdot & \cdot\\
  0 & 0 & \cdot\cdot\cdot & 1
}
\right]-\left[
\matrix{
  1 & 0 & \cdot\cdot\cdot & 0\\
  0 & 1 & \cdot\cdot\cdot & 0\\
  \cdot & \cdot & \cdot\cdot\cdot & \cdot\\
  0 & 0 & \cdot\cdot\cdot & 1
}
\right]=0
$$
也就是说，在输入的数量级很大时，梯度消失为0，造成参数更新困难。

#### 维度与点积大小的关系是怎么样的？为什么使用维度的根号来放缩？

假设向量$q$和$k$的各个分量是相互独立的随机变量，均值是0，方差是1，那么点积$q\cdot k$的均值是0，方差是$d_k$。方差越大也就说明，点积的数量级越大（以越大的概率取大值）。那么一个自然的做法就是把方差稳定到1，做法是将点积除以$\sqrt{d_k}$，这样有$D(\frac{q\cdot k}{\sqrt{d_k}})=\frac{d_k}{(\sqrt{d_k})^2}=1$。将方差控制为1，也就有效控制了前面提到的梯度消失的问题。

总结来说就是：当维度升高使得mul性注意力的整体方差变大，进而出现极大值使softmax梯度消失，所以通过scale控制方差，进而稳定梯度流，防止block堆叠时这一情况的恶化。

### Q4：为什么Transformer需要进行Multi-head Attention？

> 参考：https://www.zhihu.com/question/341222779

在我的理解中，Transformer的多头注意力借鉴了CNN中同一卷积层内使用多个卷积核的思想，原文中使用了 8 个 `scaled dot-product attention` ，在同一 `multi-head attention` 层中，输入均为 `KQV` ，同时进行注意力的计算，彼此之前参数不共享，最终将结果拼接起来，这样可以允许模型在不同的表示子空间里学习到相关的信息。

简而言之，就是希望每个注意力头，只关注最终输出序列中一个子空间，互相独立。其核心思想在于，抽取到更加丰富的特征信息。

原论文中，8个头/16个头最好，1个头最差，4个和32个头稍微差一点，但是差的不多。

**其他理解：**

原论文中说的是，将模型分为多个头，形成多个子空间，可以让模型去关注不同方面的信息。Multi-Head的作用真的是去关注“不同方面”的特征吗？

有大量的paper表明，Transformer，或Bert的特定层是有独特的功能的，底层更偏向于关注语法，顶层更偏向于关注语义。既然在同一层Transformer关注的方面是相同的，那么对该方面而言，不同的头关注点应该也是一样的。

我们首先来看看这个过程是怎样的。首先，所有的参数随机初始化，然后用相同的方法前传，在输出端得到相同的损失，用相同的方法后传，更新参数。在这一条线中，唯一不同的地方在于初始值的不同。设想一下，如果我们把同一层的所有参数（这里的参数都是指的$W^Q,W^K,W^V$ ）初始化成一样的（不同层可以不同），那么在收敛的时候，同一层的所有参数仍然是一样的，自然它们的关注模式也一样。那么，关注模型的不一样就来自于初始化的不一样。

### Q5：Soft attention v.s. Hard attention

#### 软注意力机制

1. 给定一个和任务相关的查询向量$q$，我们用注意力变量$z\in [1,N]$来表示被选择信息的索引位置，即$z=i$表示选择了第$i$个输入向量。这里采用Soft Attention的方式，即计算在给定$q$和$X$下，选择第$i$个输入向量的概率$\alpha_i$：
   $$
   \alpha_i=p(z=i|X,q)\\
   =softmax(s(x_i,q))\\
   =\frac{exp(s(x_i,q))}{\sum^N_{j=1}exps(x_j,q)}
   $$

2. 上式中$\alpha_i$称为注意力分布，$s(x_i,q)$是注意力打分函数，常见的计算方式为：

   - 加性模型：$s(x_i,q)=v^Ttanh(Wx_i+Uq)$

   - 点积模型：$s(x_i,q)=x^T_iq$

   - 缩放点积模型：$s(x_i,q)=\frac{x_i^Tq}{\sqrt{d}}$

   - 双线性模型：$s(x_i,q)=x^T_iWq$

     > 加性模型即**加性注意力机制**；点积模型即**乘性注意力机制**

   其中，$W,U,v$为可学习的网络参数，$d$为输入向量的维度。

3. 加权平均：注意力分布$\alpha_i$可以解释为在给定任务相关的查询$q$时，第$i$个输入向量受关注的程度。于是可以采用一种软性的注意力选择机制对输入信息进行汇总。
   $$
   att(X,q)=\sum^N_{i=1}\alpha_ix_i
   $$

#### 硬性注意力

1. Soft Attention选择的是所有输入向量在注意力分布下的期望。而硬性注意力只关注到某一个输入向量。
2. Hard Attention的两种实现方式：
   - 选择概率最高的一个输入向量，即$att(X,q)=x_j$，其中$j$是概率最大的输入向量的下标，即$j=argmax^N_{i=1}\alpha_i$
   - 另一种硬性注意力可以通过在注意力分布上随机采样的方式实现。
3. Hard Attention缺点在于基于最大采样或者随机采样的方式选择信息，导致最终的损失函数与注意力分布之间的函数关系不可导，因此无法在反向传播时进行训练。



### Layer Normalization & Batch  Normalization

### 相对位置编码（RPR）

### Global attention v.s. Local attention

### Hierarchical attention

### Attention over attention