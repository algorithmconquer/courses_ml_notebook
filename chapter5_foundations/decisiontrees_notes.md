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







