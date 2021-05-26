# 一、词向量基础

## 1.1 one-hot

无法表示单词含义；向量非常稀疏；向量长度等于词库长度；不需要练；

## 1.2 分布式表示

可以分为表示单词含义；向量稠密；向量长度是超参数；需要训练；

# 二、skipgram模型详解

词向量的学习通常是无监督学习，也就是不需要标注好的文本。

## 2.1 词向量核心

假设 :分布式假设

```shell
You shall know a word by the company it keep.你能通过周围词知道这个词的含义
```

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210304143400182.png" alt="image-20210304143400182" style="zoom:50%;" />

## 2.2 skipgram模型的目标函数

它的核心思想是通过中心词来预测它周围的单词。也就是说，如果我们的词向量训练比较到位，则这方面的预测能力会更强。实际上，这就是我们需要构建的目标函数。

假设：**语料库**:贪心科技 新 推出 了 人工智能 课程；window_size=1;

根据中心词预测周围词,则整个句子的概率表示为:
$$
中心词依次为:贪心科技,新,推出,了,人工智能,课程;\\
maximizeP(新|贪心科技)P(贪心科技|新)P(推出|新)P(新|推出)P(了|推出)P(推出|了)P(人工智能|了)P(了|人工智能)P(课程|人工智能)P(人工智能|课程);\\
maximize\prod_{w\in Text}\prod_{c\in Context(w)}P(c|w;\theta)=maximize\ log\prod_{w\in Text}\prod_{c\in Context(w)}P(c|w;\theta)=maximize\sum_{w\in Text}\sum_{c\in Context(w)}logP(c|w;\theta);\\
即\theta^* = argmaximize\sum_{w\in Text}\sum_{c\in Context(w)}logP(c|w;\theta);\\
注意: P(c|w;\theta)\in (0,1),希望(w,c)相似度大时,P(c|w)大些,希望(w,c)相似度小时,P(c|w;\theta)小些,\sum_{c\in Context(w)}P(c|w)=1;\\
相似度--使用内积表示,\sum_{c'\in Context(w)}P(c'|w;\theta)=1可以通过softmax;\\
P(c|w;\theta)=\frac{e^{u_c*v_w}}{\sum_{c'\in Context(w)}e^{u_{c'}*v_w}};\\
u_{c'}表示上下文词c'的词向量,v_w表示中心词w的词向量;\\
u_{c'},v_w是模型的参数;\\
$$
**注意:一个词既能作为中心词，也能作为周围词；**

因此，目标函数
$$
argmaximizeP(句子)=argmaximize\sum_{w\in Text}\sum_{c\in Context(w)}log\frac{e^{u_c*v_w}}{\sum_{c'\in Context(w)}e^{u_{c'}*v_w}};\\
其中\theta=[u_{c},v_w];\\
=argmaximize\sum_{w\in Text}\sum_{c\in Context(w)}[u_c*v_w-log(\sum_{c'\in Context(w)}e^{u_{c'}*v_w})];\\
$$
问题:使用梯度下降法优化上述目标函数，但计算$log(\sum_{c'\in Context(w)}e^{u_{c'}*v_w})$复杂度高；

解决:层序softmax,负采样；

## 2.3 负采样

**得到了SkipGram目标函数之后，发现了这个目标函数其实不好优化**。所以需要换一种方式去优化，其中比较流行的方法是使用负采样。

接下来我们从另外一个角度来推导SkipGram的目标函数。

S=w1, w2, w3, w4, w5, w6;

定义:$P(D=1|w_i, w_j)表示w_i, w_j出现在同一个上下文的概率$；

针对上面的句子S，设置window_size=1;

| word pairs | 是否是同一个上下文 | expect $P(D|w_i,w_j)$ |
| ---------- | ------------------ | --------------------- |
| $w_1,w_2$  | Yes(正样本)        | Higher                |
| $w_1,w_3$  | No                 | Lower                 |
| $w_1,w_4$  | No                 | Lower                 |
| $w_1,w_5$  | No                 | Lower                 |
| $w_1,w_6$  | No                 | Lower                 |
| $w_2,w_3$  | Yes(正样本)        | Higher                |
| $w_2,w_4$  | No                 | Lower                 |
| ......     | ......             | ......                |

训练遍历上面所有的组合(word pairs)，使得正样本的概率最大，负样本的概率最小，因此目标函数可以写为:
$$
argmax P(all\ word\ pairs)=argmax \prod_{(w, c)\in D}P(D=1|w, c;\theta)*\prod_{(w, c)\in \widetilde{D}}P(D=0|w, c;\theta);\\
条件:(w, c)相似时,P越大越好,(w, c)不相似时,P越小越好;P(D=1|w, c;\theta)+P(D=0|w, c;\theta)=1;\\
==>可以设计为P(D=1|w,c;\theta)=\frac{1}{1+exp(-u_cv_w)};\\
带入目标函数后:argmax\sum_{(w, c)\in D}logP(D=1|w, c;\theta)+\sum_{(w, c)\in \widetilde{D}}logP(D=0|w, c;\theta);\\
令\sigma(x)=\frac{1}{1+e^{-x}};\\
目标函数:argmax\sum_{(w, c)\in D}log\sigma(u_cv_w)+\sum_{(w, c)\in \widetilde{D}}log\sigma(-u_cv_w);\\
$$
当我们有了目标函数之后，剩下的过程无非就是优化并寻找最优参数了。实际上，通过优化最终得出来的最优参数就是训练出来的词向量。优化方法可采用梯度下降法或者随机梯度下降法。

上面的目标函数中负样本是所有负样本，负样本量很大；
$$
目标函数:argmax\sum_{(w, c)\in D}log\sigma(u_cv_w)+\sum_{(w, c)\in \widetilde{D}}log\sigma(-u_cv_w);\\
argmax\sum_{(w, c)\in D}log\sigma(u_cv_w)+\sum_{c'\in N(w)}log\sigma(-u_{c'}v_w);==>skipgram+负采样\\
sgd:for\ i=1,2,3,...,converge:\\
\ for\ each\ positive\ pairs\ (w, c):\\
\ update\ u_c,v_w,u_{c'};\\
N(w)表示以w为中心词采样的负样本;
$$
参数$u_c,v_w,u_{c'}$的梯度计算省略;

# 三、其他词向量技术

## 3.1  矩阵分解

例如有3个文档:

1.I like sports.

2.I like nlp.

3.I enjoy deep learning.

则v={I,like,sports,nlp,enjoy,deep,learning};|V|*|V|矩阵如下:

|              | I    | like | sports | nlp  | enjoy | deep | learning |
| ------------ | ---- | ---- | ------ | ---- | ----- | ---- | -------- |
| **I**        | 0    | 2    | 0      | 0    | 1     | 0    | 0        |
| **like**     | 2    | 0    | 1      | 1    | 0     | 0    | 0        |
| **sports**   | 0    | 1    | 0      | 0    | 0     | 0    | 0        |
| **nlp**      | 0    | 1    | 0      | 0    | 0     | 0    | 0        |
| **enjoy**    | 1    | 0    | 0      | 0    | 0     | 1    | 0        |
| **deep**     | 0    | 0    | 0      | 0    | 1     | 0    | 1        |
| **learning** | 0    | 0    | 0      | 0    | 0     | 1    | 0        |

A. 如果直接做矩阵分解,非常耗时间;

B. 矩阵分解前需要循环整个语料库,并把必要的值要统计出来;

C. 矩阵分解是全局方法,分解的过程依赖于所有的语料库;

总结:全局方法；个别改变，需要重新训练；

## 3.2 Glove(global vectors for word representation)向量

矩阵分解最大的缺点是每次分解依赖于整个矩阵，这就导致**假如有些个别文本改变了，按理来讲是需要重新训练的**，但优点是学习过程包含了全局的信息。相反，对于**SkipGram模型，由于训练发生在局部，所以训练起来效率高**，且能够很好把握局部的文本特征，但问题是它并不能从全局的视角掌握语料库的特点。所以，接下来的问题是能否把各自的都优点发挥出来?答案是设计一个融合矩阵分解和SkipGram模型的方法，这个答案其实是Glove模型。

1.记录每两个单词之间共同出现的频次(矩阵分解)；

2.设定窗口大小(skipgram)；

3.使用加权的最小二乘误差；
$$
J=\sum_{i,j=1}^{V}f(X_{ij})(w_i^T\widetilde{w_j}+b_i+\widetilde{b_j}-logX_{ij})^2
$$


## 3.3 高斯嵌入

语料库:词向量 效果 好 比 独热编码 好 好 很多

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210305112148069.png" alt="image-20210305112148069" style="zoom:50%;" />

考虑单词的置信度，通过均值，方差；

## 3.4 总结

上面学到的都是**静态词向量**；但是一个单词可能在不同的上下文中有不同的含义，对于一个单词,只学出对应的一个词向量是不够的；













