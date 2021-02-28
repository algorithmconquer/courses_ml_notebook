# 一、评分矩阵、奇异值分解与Funk-SVD

<img src="https://pic4.zhimg.com/80/v2-4bfcace1a80a5c21a386ffaa38fa15ab_1440w.jpg" alt="img" style="zoom:50%;" />

上图是一个user-item矩阵；行代表user，列代表item；

推荐系统最终的目标就是对于任意一个用户，预测出所有未评分物品的分值，并按分值从高到低的顺序将对应的物品推荐给用户。

## 1.1 特征值分解(eigendecomposition)

对于特征值分解，由于其只能作用于方阵，因此并不适合分解评分矩阵这个场景。

## 1.2 奇异值分解(Singular value decomposition,SVD)

假设矩阵M是一个$m*n$的矩阵，则一定存在一个分解$M=U\sum V^T$，其中U是$m*m$的正交矩阵，V是$n*n$的正交矩阵，$\sum$是$m*n$的对角阵，可以说是完美契合分解评分矩阵这个需求。其中，对角阵$\sum$还有一个特殊的性质，它的**所有元素都非负，且依次减小**。这个减小也特别快，在很多情况下，前10%的和就占了全部元素之和的99%以上，这就是说我们可以使用最大的k个值和对应大小的U、V矩阵来近似描述原始的评分矩阵。
$$
M=U\sum V^T;\\
1.U:m*m,\sum:m*n,V:n*n;\\
2.U:m*k,\sum:k*k,V:n*k;\\
$$
**但是实际上，这种方法存在一个致命的缺陷——奇异值分解要求矩阵是稠密的。**也就是说SVD不允许待分解矩阵中存在空白的部分，这一开始就与我们的问题所冲突了。

**为什么说“奇异值分解要求矩阵是稠密的”？**它其实是close-form的推导，所以矩阵每个值都严格对应到SVD分解后的矩阵乘积。假如矩阵是稀疏的有很多缺省值，那么这个SVD分解是否正确都令人怀疑。

当然，也可以想办法对缺失值先进行简单的填充，例如使用全局平均值。然而，即使有了补全策略，在实际应用场景下，user和item的数目往往是成千上万的，面对这样的规模传统SVD算法$O(n^3)$的时间复杂度显然是吃不消的。因此，直接使用传统SVD算法并不是一个好的选择。**既然传统SVD在实际应用场景中面临着稀疏性问题和效率问题，那么有没有办法避开稀疏问题，同时提高运算效率呢？**

## 1.3 Funk-SVD

主要思路是将原始评分矩阵$M(m*n)$分解成两个矩阵$P(m*k)$和$Q(k*n)$，同时**仅考察原始评分矩阵中有评分的项**分解结果是否准确，而判别标准则是均方差。分解后矩阵中对应的值就是:$$M_{UI}^{'}=\sum_{k=1}^KP_{Uk}Q_{kI}$$

对于整个评分矩阵而言，总的损失就是:$SSE=E^2=\sum_{U,I}(M_{U,I}-M_{U,I}^{'})^2$;

这种方法被称之为**隐语义模型（Latent factor model，LFM）**，其算法意义层面的解释为通过**隐含特征（latent factor）**将user兴趣与item特征联系起来。

<img src="https://pic2.zhimg.com/v2-ee23f5614cfa21e004c9f659e66a1e99_r.jpg" alt="preview" style="zoom:75%;" />

对于原始评分矩阵R，我们假定一共有三类隐含特征，于是将矩阵$R（3*4）$分解成用户特征矩阵$P（3*3）$与物品特征矩阵$Q（3*4）$。考察user1对item1的评分，可以认为user1对三类隐含特征class1、class2、class3的感兴趣程度分别为P11、P12、P13，而这三类隐含特征与item1相关程度则分别为Q11、Q21、Q31。

可以发现用户U对物品I最终的评分就是由各个隐含特征维度下U对I感兴趣程度的和，这里U对I的感兴趣程度则是由U对当前隐含特征的感兴趣程度乘上I与当前隐含特征相关程度来表示的。

如何求出使得SSE最小的矩阵P和Q??

## 1.4 随机梯度下降法

**梯度下降法（Gradient Descent）**是最常采用的方法之一，核心思想非常简单，沿梯度下降的方向逐步迭代。梯度是一个向量，表示的是一个函数在该点处沿梯度的方向变化最快，变化率最大，而梯度下降的方向就是指的负梯度方向。根据梯度下降法的定义，其迭代最终必然会终止于一阶导数（对于多元函数来说则是一阶偏导数）为零的点，即驻点。对于可导函数来说，其极值点一定是驻点，而驻点并不一定是极值点，还可能是鞍点。另一方面，极值点也不一定是最值点。

<img src="https://pic4.zhimg.com/80/v2-88061734b4abf2b6eefea6eb44b17bab_1440w.jpg" alt="img" style="zoom:50%;" />

上图为函数 ![[公式]](https://www.zhihu.com/equation?tex=y+%3Dx%5E%7B2%7D) 。从图中可以看出，函数唯一的驻点 （0，0）为其最小值点。

<img src="https://pic4.zhimg.com/80/v2-95ac40005c91f11df8dd72678ac261af_1440w.jpg" alt="img" style="zoom:50%;" />

上图为函数 ![[公式]](https://www.zhihu.com/equation?tex=y%3Dx%5E%7B3%7D) 。其一阶导数为 ![[公式]](https://www.zhihu.com/equation?tex=3x%5E%7B2%7D) ，**从而可知其同样有唯一驻点（0，0）。从图中可以看出，函数并没有极值点。**

<img src="https://pic1.zhimg.com/80/v2-5d02a41d0bfae484ce32fb622bde0d74_1440w.jpg" alt="img" style="zoom:50%;" />

上图为函数 ![[公式]](https://www.zhihu.com/equation?tex=y+%3D+x%5E%7B4%7D-x%5E%7B3%7D%E2%80%93x%5E%7B2%7D+%2B+x) 。从图像中可以看出，函数一共有三个驻点，包括两个极小值点和一个极大值点，其中位于最左边的极小值点是函数的最小值点。

<img src="https://pic1.zhimg.com/80/v2-4878064870feb48c0ebefe73cb916b2c_1440w.jpg" alt="img" style="zoom:50%;" />

上图为函数 ![[公式]](https://www.zhihu.com/equation?tex=z+%3Dx%5E%7B2%7D-y%5E%7B2%7D) 。其中点 （0，0，0）为其若干个鞍点中的一个。

可以看出梯度下降法在求解最小值时具有一定的局限性，用一句话概括就是，目标函数必须是凸函数。

可以发现，我们最终的目标函数是非凸的。这就意味着单纯使用梯度下降法可能会找到极大值、极小值或者鞍点。这三类点的稳定性按从小到大排列依次是极大值、鞍点、极小值，考虑实际运算中，浮点数运算都会有一定的误差，因此最终结果很大几率会落入极小值点，同时也有落入鞍点的概率，而对于极大值点，除非初始值就是极大值，否在几乎不可能到达极大值点。

随机梯度下降法主要是用来解决求和形式的优化问题，与上面需要优化的目标函数一致。其思想也很简单，既然对于求和式中每一项求梯度很麻烦，那么**干脆就随机选其中一项**计算梯度当作总的梯度来使用好了。

具体应用到上文中的目标函数$SSE=E^2=\sum_{U,I}(M_{U,I}-M_{U,I}^{'})^2=\sum_{U,I}(M_{U,I}-\sum_{k=1}^KP_{Uk}Q_{kI})^2$;

计算梯度:$\frac{\partial E^2}{\partial P_{Uk}}=-2EQ_{kI}$;$\frac{\partial E^2}{\partial Q_{kI}}=-2EP_{Uk}$;

在实际的运算中，为了P和Q中所有的值都能得到更新，一般是按照在线学习的方式选择评分矩阵中有分数的点对应的U、I来进行迭代。

值得一提的是，上面所说的各种优化都无法保证一定能找到最优解。有论文指出，单纯判断驻点是否是局部最优解就是一个NPC问题，但是也有论文指出SGD的解能大概率接近局部最优甚至全局最优。

另外，相比于利用了黑塞矩阵的牛顿迭代法，梯度下降法在方向上的选择也不是最优的。**牛顿法相当于考虑了梯度的梯度，所以相对更快。而由于其线性逼近的特性，梯度下降法在极值点附近可能出现震荡，相比之下牛顿法就没有这个问题。**但是在实际应用中，**计算黑塞矩阵的代价是非常大的，**在这里梯度下降法的优势就凸显出来了。因此，牛顿法往往应用于一些较为简单的模型，如逻辑回归。而对于稍微复杂一些的模型，梯度下降法及其各种进化版本则更受青睐。

https://zhuanlan.zhihu.com/p/34497989;

## 1.5 fm

假设有p个特征；
$$
y = w_0+\sum_{i=1}^{p}w_i*x_i+\sum_{i=1}^{p}\sum_{j=i+1}^{p}w_{ij}x_{ij};\\
y = w_0+\sum_{i=1}^{p}w_i*x_i+\sum_{i=1}^{p}\sum_{j=i+1}^{p}(x_{ij}\sum_{k=1}^{K}v_{ik}*v_{jk}); \\
y = w_0+\sum_{i=1}^{p}w_i*x_i+\sum_{k=1}^{K}(x_{ij}\sum_{i=1}^{p}\sum_{j=i+1}^{p}v_{ik}*v_{jk});\\
而\sum_{i=1}^{p}\sum_{j=i+1}^{p}v_{ik}*v_{jk}表示的是一个方阵的上三角元素；记A=\sum_{i=1}^{p}\sum_{j=i+1}^{p}v_{ik}*v_{jk};\\
则(\sum_{i=1}^{p}\sum_{j=1}^{p}v_{ik}*v_{jk})^2=2*A+\sum_{i=1}^{p}v_{ik}*v_{ik};\\
==>(\sum_{i=1}^{p}v_{ik})^2=2*A+\sum_{i=1}^{p}v_{ik}^2;\\
A=0.5*((\sum_{i=1}^{p}v_{ik})^2-\sum_{i=1}^{p}v_{ik}^2);==>和的平方-平方和\\
y = w_0+\sum_{i=1}^{p}w_i*x_i+0.5*((\sum_{i=1}^{p}v_{ik})^2-\sum_{i=1}^{p}v_{ik}^2);\\
$$
矩阵分解体现在你哪里呢？$w_{ij}=\sum_{k=1}^{K}v_{ik}*v_{jk};$

对于w而言，其中可学习的项就对应了评分矩阵中有分值的项，而其他由于数据稀疏导致难以学习的项就相当于评分矩阵中的未评分项。这样一来，不仅解决了数据稀疏性带来的二阶权重学习问题，同时对于参数规模，也从 ![[公式]](https://www.zhihu.com/equation?tex=O%28n%5E%7B2%7D%29) 级别降到了O(kn)级别。

## 1.6 贝叶斯个性化排序(Bayesian Personalized Ranking,BPR)

在BPR算法中，我们将任意用户u对应的物品进行标记，如果用户u在同时有物品i和j的时候点击了i，那么我们就得到了一个三元组$<u,i,j>$，它表示对用户u来说，i的排序要比j靠前;

BPR的算法训练流程：

输入：训练集D三元组，梯度步长$\alpha$, 正则化参数$\lambda$ ,分解矩阵维度k。
输出：模型参数，矩阵W,H;

**1. 随机初始化矩阵W,H**;

**2. 迭代更新模型参数：**

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210225194036987.png" alt="image-20210225194036987" style="zoom:50%;" />

**3.如果W,H收敛，则算法结束，输出W,H,否则回到步骤2.**

当我们拿到**W,H**后，就可以计算出每一个用户![u](https://math.jianshu.com/math?formula=u)对应的任意一个商品的排序分：

$x_{ui}^2=w_{u}h_{i}$，最终选择排序分最高的若干商品输出。

BPR是基于矩阵分解的一种排序算法，但是和funkSVD之类的算法比，它不是做全局的评分优化，而是**针对每一个用户自己的商品喜好分贝做排序优化**。因此在迭代优化的思路上完全不同。同时对于训练集的要求也是不一样的，**funkSVD只需要用户物品对应评分数据二元组做训练集，而BPR则需要用户对商品的喜好排序三元组做训练集**。

## 1.5 凸函数的判定

关于凸函数的判定，对于一元函数来说，一般是求二阶导数，若其二阶导数非负，就称之为凸函数。对于多元函数来说判定方法类似，只是从判断一元函数的单个二阶导数是否非负，变成了判断所有变量的二阶偏导数构成的**黑塞矩阵（Hessian Matrix）**是否为半正定矩阵。判断一个矩阵是否半正定可以判断所有特征值是否非负，或者判断所有主子式是否非负。

## 1.6 总结

首先因为低秩假设，一个用户可能有另外一个用户与他线性相关（物品也一样），所以用户矩阵完全可以用一个比起原始UI矩阵更低维的矩阵表示，pureSVD就可降维得到两个低维矩阵，但是此方法要求原始矩阵稠密，因此要填充矩阵（只能假设值），因此有了funkSVD直接分解得到两个低维矩阵。因为用户,物品的偏置爱好问题所以提出了biasSVD。因为用户行为不仅有评分，且有些隐反馈（点击等），所以提出了SVD++。因为假设用户爱好随时间变化，所以提出了timeSVD。因为funkSVD分解的两个矩阵有负数，现实世界中不好解释，所以提出了NMF。为了符合TopN推荐，所以提出了WMF。推翻低秩假设，提出了LLORMA（局部低秩）。因为以上问题都未解决数据稀疏和冷启动问题，所以需要用上除了评分矩阵之外的数据来使推荐更加丰满，即加边信息。