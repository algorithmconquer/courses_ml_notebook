# 一、决策树

决策树算法本身是贪心算法。所以在决策树的训练上，每一步的最优选择局限在局部；

## 1、什么是好的特征?

好的特征会减少不确定性；

不确定性--信息熵；

### 信息熵----衡量不确定性

$$
H=-\sum p_{i}logp_{i}\\
例1:arr1={0,0,1,1,0,1,1,1};H1=-\sum^2_{i=1}p_{i}logp_{i}=-(\frac{3}{8}*log\frac{3}{8}+\frac{5}{8}*log\frac{5}{8});\\
例2:arr1={0,0,0,1,0,0,0,0};H2=-(\frac{1}{8}*log\frac{1}{8}+\frac{7}{8}*log\frac{7}{8});\\
结果:H2<H1;==>数据越混乱，不确定性越大，信息熵越大；
$$

### 不确定的减少=原来的不确定性-现在的不确定性(分割后的不确定性)

不确定的减少=原来的不确定性-现在的不确定性；

现在的不确定性通过线性加权方式计算；

### 信息增益=不确定的减少

$$
IG(T,a)=H(T)-H(T|a)\\
$$

![image-20210209115218242](/Users/zenmen/Projects/courses_ml_notebook/images/image-20210209115218242.png)

原来不确定性H(T)计算
$$
11个F,5个N;\\
H(T)=-(\frac{5}{16}log\frac{5}{16}+\frac{11}{16}log\frac{11}{16});
$$

#### 单个特征信息增益计算

##### 特征Time

$$
Morning:{2F},Afternoon:{7F,4N},Night:{2F,1N};\\
H_1(T|a)=-[\frac{2}{16}*0+\frac{11}{16}*(\frac{7}{11}log\frac{7}{11}+\frac{4}{11}log\frac{4}{11})+\frac{3}{16}*(\frac{2}{3}log\frac{2}{3}+\frac{1}{3}log\frac{1}{3})];===>线性加权方式\\
G=H(T)-H_1(T|a);
$$

##### 特征MatchType

$$
Grandslam:{6F,1N},Master:{3F,3N},Friendiy:{2F,1N};\\
H_2(T|a)=-[\frac{7}{16}*(\frac{6}{7}log\frac{6}{7}+\frac{1}{7}log\frac{1}{7})+\frac{6}{16}*(\frac{3}{6}log\frac{3}{6}+\frac{3}{6}log\frac{3}{6})+\frac{3}{16}*(\frac{1}{3}log\frac{1}{3}+\frac{2}{3}log\frac{2}{3})];\\
G=H(T)-H_2(T|a);\\
$$

其他两个特征类似；

## 2、决策树的过拟合

对于决策树我们如何减少过拟合现象? 答案就是:决策树越简单越好!那什么叫更简单的决策树呢?一个重要标准是来判断决策树中节点的个数，节点个数越少说明这棵决策树就越简单。

### 2.1 什么时候可以停止分裂？

A.当一个叶节点里包含的所有样本都属于同一个类别；

B.当一个叶节点里包含所有样本的特征都相同时；

### 2.2 避免过拟合

A.设置树的最大深度；

B.当叶节点的样本个数<阈值时停止分裂；

最大深度和阈值是超参数，通过交叉验证来确定；

## 3、bagging vs boosting

共同点:都是由许多弱分类器起组成；

最大区别:bagging每个弱分类起都是过拟合，而boosting的弱分类器是欠拟合；

### 3.1 bagging--随机森林

随机森林是经典的Bagging模型，等同于同时训练了很多棵决策树，并同时用这些决策树来做决策。

模型过拟合-->不稳定--> 方差大；

例子:邀请了7位专家，而且每一位专家在决策上犯错误的概率为0.3，那他们共同决策时最终犯错误的概率为多少呢?

注意:最终犯错误是通过投票机制(少数服从多数)来决定的；例如，有4位专家犯错误，3位正确，那最终结果为错误；
$$
总共有4种情况:4位专家犯错误,5位专家犯错误,6位专家犯错误,7位专家犯错误;\\
因此p=C^0_{7}*0.3^7+C^1_{7}*0.3^6*0.7^1+C^2_{7}*0.3^5*0.7^2+C^3_{7}*0.3^4*0.7^3\\
=0.126
$$
假如我们有N个不同的模型,而且每个模型预测时的方差为$\sigma^2$,则同时使用N个模型预测时的方差为多少?$\frac{\sigma^2}{N}$;

随机森林的预测过程无非是多棵决策树共同作出决策。比如对于分类型问题，可以通过投票的方式; 对于回归问题，则可以采用平均法则。

#### 需要考虑的点

A.只有一份训练数据；B.多棵决策树要优于单棵决策树；

#### 为什么要随机?

只有多样性(Diversity)才能保证随机森林的效果(稳定性)；如何多样性?随机;

#### 随机森林中的多样性

A.训练样本的随机性(booststrap有放回采样)；B.特征选择时的随机性；

#### bagging流程

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210225154549638.png" alt="image-20210225154549638" style="zoom:50%;" />

### 3.2 boosting--gbdt,xgboost

给了一个预测问题，张三在此基础上训练出了一个模型，但效果不怎么好，误差比较大。

问题:如果我们只能接受去使用这个模型但不能做任何改变，接下来需要怎么做？==>基于残差；

提升树--基于残差训练；

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210225160044893.png" alt="image-20210225160044893" style="zoom:50%;" />

残差=真实值-预测值=[1,2,5,2,6,0,-2],下图是基于上图的残差训练；

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210225160219053.png" alt="image-20210225160219053" style="zoom:50%;" />

残差=[0,-1,1,-1,1,-1,-3],再基于这个残差训练model3;

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210225232401010.png" alt="image-20210225232401010" style="zoom:50%;" />

最终预测结果:

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210225232607310.png" alt="image-20210225232607310" style="zoom:50%;" />



# 二、xgboost

学习路径:

### 如何构造目标函数？

### 目标函数直接优化难，如何近似？

### 如何把树的结构引入目标函数？

### 仍然难优化，要不要使用贪心算法？

## 2.1 使用多棵树来预测

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210227084438709.png" alt="image-20210227084438709" style="zoom:50%;" />

假设已经训练了K棵树，则对于第i个样本的(最终)预测值为:
$$
\widehat{y}_{i}=\sum_{k=1}^{K}f_{k}(x_{i}),f_{k}\in F;\\
f_{k}(x_{i})表示第k棵树对样本x_{i}的预测值；
$$

## 2.2 目标函数的构建

目标函数:
$$
Obj = \sum_{i=1}^{n}l(y_{i},\widehat{y}_{i})+\sum_{k=1}^{K}\Omega(f_{k})=损失函数+控制复杂度;
$$

### A.树的复杂度和哪些因素有关？

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210227090033543.png" alt="image-20210227090033543" style="zoom:50%;" />

有了这个认识，后面才能知道怎么用数学公式去表示树的复杂度；

### B.additive training

$$
\widehat{y}_{i}(0)=0;
\widehat{y}_{i}(1)=f_{1}(x_{i})=\widehat{y}_{i}(0)+f_{1}(x_{i});\\
\widehat{y}_{i}(2)=f_{1}(x_{i})+f_{2}(x_{i})=\widehat{y}_{i}(1)+f_{1}(x_{i});\\
......;\\
\widehat{y}_{i}(K)=f_{1}(x_{i})+f_{2}(x_{i})+...+f_{K-1}(x_{i})+f_{K}(x_{i});\\
即\widehat{y}_{i}(K)=\sum_{j=1}^{K-1}f_{j}(x_{i})+f_{K}(x_{i})=\widehat{y}_{i}(K-1)+f_{K}(x_{i});\\
$$

假设有K棵树，则$\widehat{y}_{i}=\widehat{y}_{i}(K)$;

其实也可以通过2.1节公式推导:
$$
\widehat{y}_{i}=\sum_{k=1}^{K}f_{k}(x_{i}),f_{k}\in F;\\
\widehat{y}_{i}=\sum_{k=1}^{K-1}f_{k}(x_{i})+f_{K}(x_{i})=\widehat{y}_{i}(K-1)+f_{K}(x_{i});
$$


故目标函数又可以写为:
$$
Obj = \sum_{i=1}^{n}l(y_{i},\widehat{y}_{i})+\sum_{k=1}^{K}\Omega(f_{k});\\
Obj = \sum_{i=1}^{n}l(y_{i},\widehat{y}_{i}(K))+\sum_{k=1}^{K}\Omega(f_{k});\\
Obj = \sum_{i=1}^{n}l(y_{i},\widehat{y}_{i}(K-1)+f_{K}(x_{i}))+\sum_{k=1}^{K}\Omega(f_{k});\\
Obj = \sum_{i=1}^{n}l(y_{i},\widehat{y}_{i}(K-1)+f_{K}(x_{i}))+\sum_{j=1}^{K-1}\Omega(f_{j})+\Omega(f_{K});\\
argminObj=argmin\sum_{i=1}^{n}l(y_{i},\widehat{y}_{i}(K-1)+f_{K}(x_{i}))+\Omega(f_{K});\\
\sum_{j=1}^{K-1}\Omega(f_{j})为constant,\widehat{y}_{i}(K-1)为constant;\\
其中f_{K}(x_{i}))表示第K棵树对样本x_{i}的预测值，而\widehat{y}_{i}(K)表示前K棵树的最终预测值；
$$

## 2.3 使用泰勒级数近似目标函数

### A.泰勒级数和泰勒公式

泰勒级数在近似计算中有重要作用。

泰勒公式是一个用函数在某点的信息描述其附近取值的公式。如果函数满足一定的条件，泰勒公式可以用函数在某一点的各阶导数值做系数构建一个多项式来近似表达这个函数。

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210227133830740.png" alt="image-20210227133830740" style="zoom:50%;" />

则
$$
令上面的公式x-x_{0}=\Delta x即x=x_{0}+\Delta x;\\
f(x_{0}+\Delta x)=f(x_{0})+f'(x_{0})\Delta x+\frac{f"(x_{0})}{2!}\Delta x^2+...;\\
or:f(x+\Delta x)=f(x)+f'(x)\Delta x+\frac{f"(x)}{2!}\Delta x^2+...;\\
$$

### B.对目标函数使用泰勒级数近似

$$
对于l(y_{i},\widehat{y}_{i}(K-1)+f_{K}(x_{i}))，令\Delta x=f_{K}(x_{i});\\
则l(y_{i},\widehat{y}_{i}(K-1)+f_{K}(x_{i}))=l(y_{i},\widehat{y}_{i}(K-1))+\frac{\partial l}{\partial \widehat{y}_{i}(K-1)}f_{K}(x_{i})+\frac{1}{2}\frac{\partial^2 l}{\partial \widehat{y}_{i}(K-1)}f_{K}^2(x_{i});\\
l(y_{i},\widehat{y}_{i}(K-1)+f_{K}(x_{i}))=l(y_{i},\widehat{y}_{i}(K-1))+g_{i}f_{K}(x_{i})+\frac{1}{2}h_{i}f_{K}^2(x_{i});\\
其中g_{i}=\frac{\partial l}{\partial \widehat{y}_{i}(K-1)},h_{i}=\frac{\partial^2 l}{\partial \widehat{y}_{i}(K-1)};\\
$$

### C.将泰勒级数带入并最小化目标函数

$$
minimize Obj=minimize \sum_{i=1}^{n}l(y_{i},\widehat{y}_{i}(K-1)+f_{K}(x_{i}))+\Omega(f_{K});\\
=minimize\sum_{i=1}^{n}[l(y_{i},\widehat{y}_{i}(K-1))+g_{i}f_{K}(x_{i})+\frac{1}{2}h_{i}f_{K}^2(x_{i})]+\Omega(f_{K});\\
=minimize\sum_{i=1}^{n}[g_{i}f_{K}(x_{i})+\frac{1}{2}h_{i}f_{K}^2(x_{i})]+\Omega(f_{K});\\
注: 当训练第K棵树时，(g_{i},h_{i})是已知的;\\
$$

$(g_{i},h_{i})$作用和价值在哪呢？

$(g_{i},h_{i})$代表了残差，训练完K-1棵树时，这些残差表现在了$(g_{i},h_{i})$上的，$(g_{i},h_{i})$传递信息；

## 2.4 新的目标函数(假设树的形状已知)

$f_K(x_i)$的表达式是什么？

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210228100846077.png" alt="image-20210228100846077" style="zoom:50%;" />

注意:
$$
q(x)--样本x的位置；w--叶子节点值组成的向量；f_k(x_i)--样本x_{i}的预测值；I_j--第j个叶子节点的样本集合；
$$
$\Omega(f_{K})$的表达式是什么？

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210228110746162.png" alt="image-20210228110746162" style="zoom:50%;" />

带入目标函数：
$$
minimize\sum_{i=1}^{n}[g_{i}f_{K}(x_{i})+\frac{1}{2}h_{i}f_{K}^2(x_{i})]+\Omega(f_{K});\\
minimize\sum_{i=1}^{n}[g_{i}w_{q(x_i)}+\frac{1}{2}h_{i}w^2_{q(x_i)}]+\gamma T+\frac{1}{2}\lambda \sum_{j=1}^{T}w^2_{j};按照样本组织\\
minimize\sum_{j=1}^{T}[(\sum_{i\in I_j}g_{i})w_j+\frac{1}{2}(\sum_{i\in I_j}h^2_{j})w^2_j]+\gamma T+\frac{1}{2}\lambda \sum_{j=1}^{T}w^2_{j};按照叶节点组织\\
minimize\sum_{j=1}^{T}[(\sum_{i\in I_j}g_{i})w_j+\frac{1}{2}(\sum_{i\in I_j}h^2_{j}+\lambda)w^2_j]+\gamma T;\\
上式看作ax^2+bx+c的二次函数,并且令G_j=\sum_{i\in I_j}g_{i},H_j=\sum_{i\in I_j}h^2_{j};\\
则w^*_j=-\frac{G_i}{2*\frac{1}{2}(H_i+\lambda)}=-\frac{G_i}{H_i+\lambda};\\
obj^*=\sum_{j=1}^{T}(-\frac{G^2_{j}}{H_i+\lambda}+0.5(H_j+\lambda)\frac{G^2_{j}}{(H_i+\lambda)^2})+\gamma T;\\
obj^*=-0.5\sum_{j=1}^{T}\frac{G^2_{j}}{H_i+\lambda}+\gamma T;\\
$$
Xxxxx========

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210228130142531.png" alt="image-20210228130142531" style="zoom:50%;" />



## 2.5 如何寻找树的形状

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210228131901206.png" alt="image-20210228131901206" style="zoom:50%;" />

决策树:使用信息增益选择最好特征；

Xgboost:使用obj选择最好特征；

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210228132734119.png" alt="image-20210228132734119" style="zoom:50%;" />

## ABC



