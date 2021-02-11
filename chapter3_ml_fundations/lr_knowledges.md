# 一、分类问题

学习从输入到输出的映射$f:x-->y$;

# 二、分类问题

假设有办法表示条件概率$P(y=0|x)和P(y=1|x)$,怎么分类呢？

if $P(y=0|x) > P(y=1|x)$:

​	$y=0$

else:

​	$y=1$

这里的核心问题是:通过条件概率$p(y|x)$来描述$x$和$y$之间的关系。逻辑回归实际上是基于线性回归模型构建起来的；

## 2.1 怎么通过线性回归方程怎么构造$p(y|x)$?

现在需要寻找$(-\infty, +\infty)$到$(0, 1)$之间的映射:

其中$p(y=1|x),p(y=0|x)\in(0,1)$且$p(y=1|x)+p(y=0|x)=1$;
$$
线性模型w^Tx+b;和逻辑函数sigmoid(x);结合
$$


因此
$$
p(y=1|x;w,b)=\frac{1}{1+e^{-(w^Tx+b)}}=(\frac{1}{1+e^{-(w^Tx+b)}})^1+(1-\frac{1}{1+e^{-(w^Tx+b)}})^0;\\
p(y=0|x;w,b)=\frac{e^{-(w^Tx+b)}}{1+e^{-(w^Tx+b)}}=(\frac{1}{1+e^{-(w^Tx+b)}})^0+(1-\frac{1}{1+e^{-(w^Tx+b)}})^1;\\
合并后p(y|x;w,b)=(\frac{1}{1+e^{-(w^Tx+b)}})^y+(1-\frac{1}{1+e^{-(w^Tx+b)}})^{(1-y)};\\
$$

## 2.2 lr的目标函数

这个目标函数实际上是从条件概率获得的;

### 1、最大似然估计

假如有个未知的模型(看作是黑盒子)，并且它产生了很多能观测到的样本。这时候，我们便可以通过最大化这些样本的概率反过来求出模型的最优参数，这个过程称之为最大似然估计。

核心:根据观测到的结果来估计未知的参数；

例子:假设未知参数为$\theta$, 已知样本为$D$,最大似然估计通过最大化$P(D|\theta)$来求解未知参数$\theta$;

$D=\{H,T,T,H,H,H\};P(H)=\theta,\theta=?$

计算
$$
argmax\ P(D|\theta)=argnax\ P(H,T,T,H,H,H|\theta)=argmax\ P(H)^4P(T)^2=argmax\ \theta^4(1-\theta)^2;\\
其中样本之间独立,\theta\in(0,1),P(D|\theta)是似然概率;\\
$$
求解
$$
\frac{\partial L}{\partial \theta}=4\theta^3*(1-\theta)^2+\theta^4*2(1-\theta)*(-1)=0;\\
\theta^3(1-\theta)[2(1-\theta)-\theta]=\theta^3(1-\theta)(2-3\theta)=0;\\
可行解\theta=\frac{2}{3};
$$

### 2、每一个样本的似然概率

数据集$D=\{(x_i, y_i)\}^N_{i=1},x_i\in R^d,y_i\in \{0,1\}$;

每一个样本的似然概率怎么表示呢?
$$
记p_i=p(y_i=1|x_i;w,b)=\frac{1}{1+e^{-(w^Tx_i+b)}};\\
p(y_i|x_i;w,b)=p_i^{y_i}(1-p_i)^{(1-y_{i})}=(\frac{1}{1+e^{-(w^Tx_i+b)}})^{y_i}(1-\frac{1}{1+e^{-(w^Tx_i+b)}})^{(1-y_{i})};\\
$$
所有样本的似然概率怎么表示呢 ？
$$
P(D|w,b)=\prod^{N}_{i=1}p(y_i|x_i;w,b)=\prod^{N}_{i=1}p_i^{y_i}(1-p_i)^{(1-y_{i})};\\
$$
有了所有样本的似然概率之后，我们的目标就是要求出让这个似然概率最大化的模型的参数(对于逻辑回归模型就是$w,b$)。这个过程称之为最大似然估计(maximum likelihood estimation)。

### 3、lr的最大似然估计

$$
\widehat{w}_{MLE},\widehat{b}_{MLE}=argmaxP(D|w,b)=argmax\prod^{N}_{i=1}p(y_i|x_i;w,b)=argmax\prod^{N}_{i=1}p_i^{y_i}(1-p_i)^{(1-y_{i})};\\
=argmaxlog\prod^{N}_{i=1}p_i^{y_i}(1-p_i)^{(1-y_{i})};\\
=argmax\sum^{N}_{i=1}[y_ilogp_i+(1-y_i)log(1-p_i)];\\
=argmin-\sum^{N}_{i=1}[y_ilogp_i+(1-y_i)log(1-p_i)];\\
$$

### 4、lr的梯度下降法

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210211104140284.png" alt="image-20210211104140284" style="zoom:50%;" />





