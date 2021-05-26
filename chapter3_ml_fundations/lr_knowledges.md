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

# 三、lr的一切

## 3.1 从回归到分类

怎么将前面的y值连续的线性回归模型应用到分类上去？

最简单的是使用unit-step函数
$$
y=\begin{cases} 0,z<0\\ 0.5,z=0, \\ 1,z>0\end{cases}
$$
但是这个函数不连续；

改进:使用sigmoid函数；

### sigmoid函数起到什么作用呢?

1、线性回归是在实数范围内预测，而分类范围需要把y值限制在$(0,1)$;因此sigmoid为lr减少了预测范围；

2、线性回归的敏感度是一致的，而sigmoid是在0附近更为敏感，更加关注分类边界，增加了模型的鲁棒性；

## 3.2 lr模型的假设

A、假设数据服从伯努利分布；

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210214092751236.png" alt="image-20210214092751236" style="zoom:50%;" />

B、假设$p(y=1|x;w,b)=sigmoid(w^Tx+b)$;

## 3.3 lr模型主要解决什么问题以及目的

lr的本质:lr假设数据服从伯努利分布，通过极大化似然函数的方法，利用梯度下降法来求解参数，达到二分类的目的；

## 3.4 为什么使用极大似然估计作为损失函数？

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210215101654894.png" alt="image-20210215101654894" style="zoom:50%;" />

梯度计算

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210215101957418.png" alt="image-20210215101957418" style="zoom:50%;" />

## 3.5 lr的求解方法

## 3.6 多分类如何解决

OVO,OVR,softmax;

## 3.7 特征选择

A.将高度相关的特征去掉；

B.不能进行特征选择;

## 3.8 为什么要进行特征离散化和特征交叉？

A.计算速度更快；

B.对异常值更具鲁棒性；

C.离散化相当于非线性，提升表达能力；

D.离散后可以进行特征交叉，提升表达能力；

## 3.9 lr的特征系数的绝对值可以认为是特征重要性吗？

lr的特征系数的绝对值越大，对分类效果的影响越显著，然而并不能简单认为特征系数更大的特征更重要；

A.改变变量的尺度就会改变系数绝对值；

B.如果特征是线性相关的，则系数可以从一个特征转移到另一个特征；

## 3.10 归一化对lr有什么作用?

利于分析；利于梯度下降；

# 四、正则与先验关系

最大似然估计(MLE,极大似然估计)的核心思想是给定模型参数$\theta$的前提下,通过最大化观测样本的概率来寻求最优解；

数学角度讲，最大似然估计是计算$P(D|\theta)$,其中$D$表示观测到的样本，$\theta$表示模型的参数;

## 4.1 MLE vs MAP

最大后验估计(MAP,最大化后验概率)计算的是$P(\theta|D)$;

$\theta_{MLE}=argmaxP(D|\theta)$;$\theta_{MAP}=argmaxP(\theta|D)$;

联系:
$$
\theta_{MAP}=argmaxP(\theta|D)=argmax\frac{P(D|\theta)P(\theta)}{P(D)};\\
其中P(D|\theta)是似然概率，P(\theta)是先验概率，P(D)是归一化概率；\\
数学角度看:加入P(\theta)相当于加入正则；直观看；
$$
观测样本少时，很容易被我们看到的样本所迷惑，这时应加入先验概率，先验概率起的作用大；随着观测样本增多，通过最大似然估计出来的参数越来越准确，这时加入加入先验概率起的作用小；
当**数据量无穷多**的时候，最大后验估计的结果会逼近于最大似然估计的结果。这就说明，当数据越来越多的时候，先验的作用会逐步减弱。

## 4.2 先验与正则

MAP要比MLE多出了一个项，就是先验$p(\theta)$;这个概率分布是我们可以提前设定好的;可以通过先验的方式来给模型灌输一些经验信息。比如设定参数$\theta$服从高斯分布或者其他类型的分布。不同先验分布和正则之间也有着很有趣的关系，比如参数的先验分布为高斯分布时，对应的为L2正则;当参数的先验分布为拉普拉斯分布时，对应的为L1正则。

### 4.2.1 参数的先验分布服从高斯分布

$$
P(D|\theta)=\prod_{i=1}^{N}p(y_i|x_i;\theta);\\
\theta^{*}_{MLE}=argmaxP(D|\theta)=argmx\prod_{i=1}^{N}p(y_i|x_i;\theta)=argmaxlog\prod_{i=1}^{N}p(y_i|x_i;\theta)=argmax\sum_{i=1}^{N}logp(y_i|x_i;\theta);\\
假设P(\theta)\sim N(0,\sigma^2)==>p(\theta)=e^{-\frac{\theta^2}{2\sigma^2}};\\
$$

现在计算最大后验概率估计
$$
\theta^{*}_{MAP}=argmaxP(\theta|D)=argmax\frac{P(D|\theta)P(\theta)}{P(D)}=argmaxP(D|\theta)P(\theta)=argmaxlogP(D|\theta)P(\theta)=argmaxlogP(D|\theta)+logP(\theta);\\
\theta^{*}_{MAP}=argmax\sum_{i=1}^{N}logp(y_i|x_i;\theta)+log\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{\theta^2}{2\sigma^2}}=argmax\sum_{i=1}^{N}logp(y_i|x_i;\theta)+log\frac{1}{\sqrt{2\pi}\sigma}+loge^{-\frac{\theta^2}{2\sigma^2}};\\
\theta^{*}_{MAP}=argmin-(\sum_{i=1}^{N}logp(y_i|x_i;\theta)+const-\frac{\theta^2}{2\sigma^2})=argmin-\sum_{i=1}^{N}logp(y_i|x_i;\theta)+\frac{1}{2\sigma^2}\theta^2;\\
其中\frac{1}{2\sigma^2}\theta^2相当于正则项,\lambda=\frac{1}{2\sigma^2};\\
$$

### 4.2.2 参数的先验分布服从拉普拉斯分布

$$
假设P(\theta)\sim Laplace(\mu, b);==>设\mu=0,P(\theta)\sim Laplace(0, b);==>p(\theta)=\frac{1}{2b}e^{-\frac{|\theta|}{b}};\\
\theta^{*}_{MAP}=argmaxlogP(D|\theta)P(\theta)=argmaxlogP(D|\theta)+logP(\theta)=argmax\sum_{i=1}^{N}logp(y_i|x_i;\theta)+log\frac{1}{2b}e^{-\frac{|\theta|}{b}};\\
\theta^{*}_{MAP}=argmin-(\sum_{i=1}^{N}logp(y_i|x_i;\theta)+log\frac{1}{2b}-\frac{|\theta|}{b})=argmin-\sum_{i=1}^{N}logp(y_i|x_i;\theta)+const+\frac{1}{b}|\theta|;\\
\theta^{*}_{MAP}=argmin-\sum_{i=1}^{N}logp(y_i|x_i;\theta)+\frac{1}{b}|\theta|;
$$

### 4.2.3 总结：最大后验估计与正则

假设参数的先验概率服从高斯分布，相当于加入了L2正则；

假设参数的先验概率服从拉普拉斯分布，相当于加入了L1正则；

# 五、dm

missing_data处理:1、扔掉；2、众数，中位数，平均数；3、单独看作一个值；4、提前predict；









