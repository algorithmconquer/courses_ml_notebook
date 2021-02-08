# 一、朴素贝叶斯
## 1、应用
垃圾邮件分类，情感分析，文本主题分类；
## 2、朴素贝叶斯的两个阶段

A.计算每个单词在不同分类中所出现的概率，这个概率是基于语料库(训练数据)来获得的。
B.利用已经计算好的概率，再结合贝叶斯定理就可以算出对于一个新的文本，它属于某一个类别的概率值，并通过这个结果做最后的分类决策。

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210208151445180.png" alt="image-20210208151445180" style="zoom:67%;" />

step1:计算单词出现的概率(训练阶段)


$$
P("购买"|"正常")=\frac{3}{24*10};
P("购买"|"垃圾")=\frac{7}{12*10};
$$
step2:先验概率(训练阶段)
$$
P("正常")=\frac{24}{24+12};P("垃圾")=\frac{12}{24+12};
$$
step3:预测阶段P("正常"|"邮件内容")，P("垃圾"|"邮件内容")

需要用到一个著名的概率公式-贝叶斯定理。利用贝叶斯定理可以把上述条件概率做进一步的分解，最终可以计算出它们的值；
$$
贝叶斯定理:P(y|x)=\frac{P(x|y)*P(y)}{P(x)};\\
P(y|x)--后验概率；P(x|y)--似然概率；P(y)--先验概率；P(x)--归一化概率
$$
![image-20210208155924133](/Users/zenmen/Projects/courses_ml_notebook/images/image-20210208155924133.png)

通过上图预测P(垃圾|邮件内容),P(正常|邮件内容):
$$
\begin{split}P(垃圾|邮件内容)=\frac{P(邮件内容|垃圾)P(垃圾)}{P(邮件内容)}=\frac{P(购买,物品,不是,广告,这|垃圾)P(垃圾)}{P(邮件内容)}\\
=\frac{P(购买|垃圾)P(物品|垃圾)P(不是|垃圾)P(广告|垃圾)P(这|垃圾)P(垃圾)}{P(邮件内容)}\\
\end{split}
$$
注意:上面使用了条件独立假设；
$$
条件独立:P(x1,x2,x3|y)=P(x1|y)*P(x2|y)*P(x3|y)
$$

## 3、贝叶斯平滑

### 3.1 加1平滑(add one smoothing)

$$
P(w|y=c)=\frac{类别为c的语料库中单词w出现的次数+1}{类别为c的语料库中包含所有单词的次数之和+v}
$$

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210208164243792.png" alt="image-20210208164243792" style="zoom:67%;" />

注意: 计算P(垃圾|邮件内容)时可能会出现underflow;通过对概率取对数来解决;

## 4、朴素贝叶斯的最大似然估计

最大似然估计:根据观测到的样本预测未知参数theta;

![image-20210208193957097](/Users/zenmen/Projects/courses_ml_notebook/images/image-20210208193957097.png)

sample1:今天 上午 下雨, y=1;sample2:明天 上午 上午, y=2;sample3:下雨 时 在家; y=3;

词库:{今天, 上午, 下雨, 明天, 时, 在家};|V|=6;

则sample1:([1,1,1,0,0,0], 1);samples=2:([0,2,0,1,0,0], 2);sample3:([0,0,1,0,1,1],3);
$$
D={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)};\\
最大似然概率=argmax P(D)= argmx \prod^N_{i=1}P(x_i, y_i)=argmax \prod^N_{i=1}P(y_i)*P(x_i|y_i)\\
注意:和lr模型求P(y_i=1|x_i)对比,lr模型是判别模型(计算条件概率),而贝叶斯是生成模型(计算联合概率);
$$
现在表示sample2的似然概率:
$$
P(x_2|y_2)=P(明天,上午,上午|y_2)=P(明天|y_2)*P(上午|y_2)*P(上午|y_2)\\
=P(明天|y_2)^1*P(上午|y_2)^2\\
=P(今天|y_2)^0*P(上午|y_2)^2*P(下雨|y_2)^0*P(明天|y_2)^1*P(时|y_2)^0*P(在家|y_2)^0\\
=\prod^{V}_{j=1}P(w_j|y_i)^{x_{ij}}
$$
因此
$$
argmaxP(D)=argmax\prod^{N}_{i=1}(P(y_i)*\prod^{V}_{j=1}P(w_j|y_i)^{x_{ij}})\\
=argmaxlog\prod^{N}_{i=1}(P(y_i)*\prod^{V}_{j=1}P(w_j|y_i)^{x_{ij}})\\
=argmax\sum^{N}_{i=1}[logP(y_i)+\sum^V_{j=1}x_{ij}logP(w_j|y_i)]
$$
怎么引入参数$\theta$?

假设类别数为K,词库大小为V；
$$
参数\theta=\left[
	\begin{matrix}
		\theta_{11} & \theta_{12} & \cdots & \theta_{1K}\\
		\theta_{21} & \theta_{22} & \cdots & \theta_{2K}\\
		\vdots & \vdots & \ddots & \vdots \\
		\theta_{V1} & \theta_{V2} & \cdots & \theta_{VK}\\
	\end{matrix}
\right]
$$
先验概率\pi
$$
\pi = [\pi_{1}, \pi_{2}, ..., \pi_{K}]
$$
带入最大似然概率
$$
argmaxP(D)=argmax\sum^K_{k=1}\sum^M_{i:y_i=k}[logP(y_i=k)+\sum^V_{j=1}x_{ij}logP(w_j|y_i=K)]\\
=argmax\sum^K_{k=1}[\sum^M_{i:y_i=k}logP(y_i=k)+\sum^M_{i:y_i=k}\sum^V_{j=1}x_{ij}logP(w_j|y_i=K)]\\
=argmax\sum^K_{k=1}[n_k*log\pi_{k}+\sum^M_{i:y_i=k}\sum^V_{j=1}x_{ij}*log\theta_{jk}]\\
=argmax\sum^K_{k=1}n_k*log\pi_{k}+\sum^K_{k=1}\sum^M_{i:y_i=k}\sum^V_{j=1}x_{ij}*log\theta_{jk}
$$
上面公式是有约束的
$$
\sum^K_{k=1}\pi_{k}=1;\\
\sum^V_{j=1}\theta_{jk}=1;k=1,..,K;也就是有K个约束
n_k表示属于类别k的文档数;
$$

### 4.1 带约束条件的优化--拉格朗日乘子法

例
$$
f(x, y) = x + y;\\
s.t\ x^2+y^2=1;\\
解:
L(x,y,\lambda) = f(x, y)+\lambda(x^2+y^2-1)\\
对x,y,\lambda分别求导,然后解3元1次方程;
$$

### 4.2 朴素贝叶斯目标函数的优化

$$
argmax\sum^K_{k=1}n_k*log\pi_{k}+\sum^K_{k=1}\sum^M_{i:y_i=k}\sum^V_{j=1}x_{ij}*log\theta_{jk}\\
s.t\ \sum^K_{k=1}\pi_{k}=1;\sum^V_{j=1}\theta_{jk}=1;k=1,..,K;\\
拉格朗日乘子法:\\
argmax\sum^K_{k=1}n_k*log\pi_{k}+\sum^K_{k=1}\sum^M_{i:y_i=k}\sum^V_{j=1}x_{ij}*log\theta_{jk}+\lambda_1*(sum^K_{u=1}\pi_{u}-1)+\sum^K_{k=1}\lambda_k*(\sum^V_{v=1}\theta_{vk}-1)\\
=argmax L
$$

求解参数
$$
\frac{\partial L}{\partial \pi_{k}}=\frac{n_k}{\pi_{k}}+\lambda_{1}=0;==>\pi_{k}=-\frac{n_k}{\lambda_{1}}\\
带入约束条件:\sum^K_{k=1}\pi_{k}=\sum^K_{k=1}(-\frac{n_k}{\lambda_{1}})=1;==>\lambda_{1}=-\sum^K_{k=1}n_{k};
则\pi_{k}=-\frac{n_k}{\lambda_{1}}=\frac{n_k}{\sum^K_{k=1}n_{k}};
$$
求解参数
$$
对j,k变量求导时,i为常量;\\
\frac{\partial L}{\theta_{jk}}=\sum^M_{i:y_i=k}\frac{x_{ij}}{\theta_{jk}}+\lambda_{k};==>\theta_{jk}=-\lambda_{k}\sum^M_{i:y_i=k}x_{ij};\\
带入约束条件得到:\lambda_{k}=-\frac{1}{\sum^{V}_{j=1}\sum^M_{i:y_i=k}x_{ij}};\\
则\theta_{jk}=\frac{\sum^M_{i:y_i=k}x_{ij}}{\sum^{V}_{j=1}\sum^M_{i:y_i=k}x_{ij}};
$$




