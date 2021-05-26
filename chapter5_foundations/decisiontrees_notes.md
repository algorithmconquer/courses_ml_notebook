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

## 1.1 学习路径:

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

## 3. 训练好的模型解析

### 3.1 训练好的模型文件(以一棵树为例)

```shell
booster[0]:
0:[u_demand_ck_LEISURE<0.071999] yes=1,no=2,missing=1,gain=879772.0,cover=29654000.0
    1:[sim_u_clk6h_topic64-d_topics64<2.0] yes=3,no=4,missing=4,gain=179492.0,cover=22877700.0
        3:[uvd_ctravg<0.403999] yes=7,no=8,missing=7,gain=33540.9,cover=3852110.0
            7:[sulsd_ctravg<0.202999] yes=15,no=16,missing=15,gain=18954.8,cover=3346250.0
                15:[avg_u_clk6h_tag-d_tags<0.999999] yes=31,no=32,missing=31,gain=7604.07,cover=2186700.0
                    31:[sud_ctravg<0.099999] yes=63,no=64,missing=64,gain=3845.73,cover=1563190.0
                        63:[ud_ctravg<0.084999] yes=127,no=128,missing=127,gain=1045.89,cover=570000.0
                            127:leaf=-0.56681,cover=337756.0
                            128:leaf=-0.540614,cover=232245.0
                        64:[ud_ctravg<0.114999] yes=129,no=130,missing=129,gain=1705.81,cover=993190.0
                            129:leaf=-0.533808,cover=672228.0
                            130:leaf=-0.507201,cover=320962.0
                    32:[uvd_ctravg<0.197999] yes=65,no=66,missing=66,gain=3245.24,cover=623512.0
                        65:[max_u_clk5m_tag-d_tags<2.0] yes=131,no=132,missing=131,gain=977.591,cover=387474.0
                            131:leaf=-0.520899,cover=323579.0
                            132:leaf=-0.480232,cover=63895.0
                        66:[max_u_clk5m_tag-d_tags<2.0] yes=133,no=134,missing=133,gain=660.345,cover=236038.0
                            133:leaf=-0.477689,cover=187029.0
                            134:leaf=-0.438496,cover=49008.2
                16:[max_u_clk6h_ct-d_ct<2.68] yes=33,no=34,missing=33,gain=8339.1,cover=1159540.0
                    33:[avg_u_demand_ck-d_demand<0.039999] yes=67,no=68,missing=68,gain=3558.28,cover=828921.0
                        67:[u_demand_ck_LEISURE<-0.812001] yes=135,no=136,missing=136,gain=844.773,cover=426857.0
                            135:leaf=-0.532739,cover=134342.0
                            136:leaf=-0.503952,cover=292515.0
                        68:[newsType<14.0] yes=137,no=138,missing=138,gain=1258.36,cover=402064.0
                            137:leaf=-0.467347,cover=352203.0
                            138:leaf=-0.518318,cover=49860.8
                    34:[sud_ctravg<0.315999] yes=69,no=70,missing=69,gain=2939.07,cover=330624.0
                        69:[ud_ctrmin<0.064999] yes=139,no=140,missing=140,gain=850.916,cover=171725.0
                            139:leaf=-0.481375,cover=106304.0
                            140:leaf=-0.437828,cover=65420.2
                        70:[max_u_clk30m_ct-d_ct<8.0] yes=141,no=142,missing=141,gain=1953.81,cover=158899.0
                            141:leaf=-0.421902,cover=135733.0
                            142:leaf=-0.327592,cover=23165.8
            8:[avg_u_vtags_ck-d_tags<1.019] yes=17,no=18,missing=17,gain=8079.26,cover=505866.0
                17:[max_u_clk6h_tag-d_tags<0.999999] yes=35,no=36,missing=35,gain=2549.36,cover=374745.0
                    35:[d_stats_clickNum<277.0] yes=71,no=72,missing=72,gain=1329.3,cover=225814.0
                        71:[d_ext_h7<2.0] yes=143,no=144,missing=144,gain=101.159,cover=20876.0
                            143:leaf=-0.510044,cover=6769.0
                            144:leaf=-0.555366,cover=14107.0
                        72:[sud_ctravg<0.215999] yes=145,no=146,missing=146,gain=1072.51,cover=204938.0
                            145:leaf=-0.502924,cover=43644.2
                            146:leaf=-0.449856,cover=161294.0
                    36:[u_topics512_ck_166<-1.551] yes=73,no=74,missing=74,gain=1040.95,cover=148931.0
                        73:[u_topics512_cs_183<0.926999] yes=147,no=148,missing=147,gain=207.959,cover=162.0
                            147:leaf=-0.36,cover=30.25
                            148:leaf=0.502825,cover=131.75
                        74:[sim_u_vtopics64_cs-d_topics64<-1e-06] yes=149,no=150,missing=149,gain=1036.2,cover=148769.0
                            149:leaf=-0.522712,cover=8169.75
                            150:leaf=-0.412714,cover=140599.0
                18:[uvd_ctravg<0.607999] yes=37,no=38,missing=38,gain=2129.9,cover=131120.0
                    37:[sim_u_clk6h_tag-d_tags<0.318999] yes=75,no=76,missing=75,gain=577.086,cover=71780.8
                        75:[sim_u_vtopics64_ck-d_topics64<0.049999] yes=151,no=152,missing=151,gain=205.921,cover=43223.2
                            151:leaf=-0.502184,cover=2517.0
                            152:leaf=-0.413363,cover=40706.2
                        76:[sim_u_vtopics512_ck-d_topics512<-0.105001] yes=153,no=154,missing=153,gain=412.338,cover=28557.5
                            153:leaf=-0.534955,cover=1212.0
                            154:leaf=-0.355888,cover=27345.5
                    38:[sim_u_vtopics128_ck-d_topics128<0.326999] yes=77,no=78,missing=78,gain=1013.32,cover=59339.8
                        77:[d_stats_rawctr<0.139999] yes=155,no=156,missing=156,gain=643.712,cover=53180.5
                            155:leaf=-0.327496,cover=37826.0
                            156:leaf=-0.254606,cover=15354.5
                        78:[u_topics128_ck_43<-0.314001] yes=157,no=158,missing=158,gain=1362.94,cover=6159.25
                            157:leaf=0.283315,cover=228.25
                            158:leaf=-0.462744,cover=5931.0
        4:[avg_u_vdemand_ck-d_demand<0.104999] yes=9,no=10,missing=9,gain=95620.0,cover=19025600.0
            9:[sim_u_seg_tags_ck-d_tags<2.0] yes=19,no=20,missing=20,gain=21683.0,cover=17112900.0
                19:[d_retrieve_als<2.0] yes=39,no=40,missing=40,gain=10041.5,cover=5009620.0
                    39:[ud_ctravg<0.114999] yes=79,no=80,missing=80,gain=3858.78,cover=619510.0
                        79:[sim_u_st_ls_topics64_cs-d_topics64<2.0] yes=159,no=160,missing=160,gain=726.826,cover=339182.0
                            159:leaf=-0.492763,cover=22261.8
                            160:leaf=-0.548967,cover=316920.0
                        80:[max_u_st_ls_topics64_cs-d_topics64<0.999999] yes=161,no=162,missing=162,gain=2031.75,cover=280328.0
                            161:leaf=-0.407229,cover=20720.5
                            162:leaf=-0.504911,cover=259607.0
                    40:[max_u_st_topics128_cs-d_topics128<0.999999] yes=81,no=82,missing=82,gain=5718.92,cover=4390110.0
                        81:[min_u_st_ls_sct_cs-d_sct<0.929999] yes=163,no=164,missing=164,gain=1386.01,cover=232038.0
                            163:leaf=-0.452892,cover=25670.0
                            164:leaf=-0.526889,cover=206368.0
                        82:[avg_u_vdemand_ck-d_demand<-0.737001] yes=165,no=166,missing=166,gain=3835.16,cover=4158080.0
                            165:leaf=-0.578519,cover=1620590.0
                            166:leaf=-0.559829,cover=2537480.0
                20:[uvd_ctravg<0.079999] yes=41,no=42,missing=42,gain=6456.14,cover=12103300.0
                    41:[avg_u_vdemand_ck-d_demand<-1.784] yes=83,no=84,missing=84,gain=711.592,cover=7344450.0
                        83:[avg_u_vdemand_ck-d_demand<-2.317] yes=167,no=168,missing=168,gain=78.828,cover=3034220.0
                            167:leaf=-0.593654,cover=1516520.0
                            168:leaf=-0.590521,cover=1517700.0
                        84:[d_retrieve_itemcf_sixhour<2.0] yes=169,no=170,missing=170,gain=377.813,cover=4310240.0
                            169:leaf=-0.571243,cover=150691.0
                            170:leaf=-0.586611,cover=4159550.0
                    42:[d_retrieve_vitemcf<2.0] yes=85,no=86,missing=86,gain=2655.82,cover=4758850.0
                        85:[avg_u_vdemand_ck-d_demand<0.104999] yes=171,no=172,missing=172,gain=408.74,cover=261791.0
                            171:leaf=-0.552788,cover=182921.0
                            172:leaf=-0.526848,cover=78870.0
                        86:[d_retrieve_vals<2.0] yes=173,no=174,missing=174,gain=1077.49,cover=4497060.0
                            173:leaf=-0.555543,cover=219463.0
                            174:leaf=-0.577134,cover=4277600.0
            10:[max_u_vkeywords_ck-d_titlekws<0.523999] yes=21,no=22,missing=21,gain=15067.2,cover=1912690.0
                21:[max_u_vtags_ck-d_tags<0.608999] yes=43,no=44,missing=43,gain=3218.03,cover=1354270.0
                    43:[sim_u_seg_subcats_ck-d_sct<2.0] yes=87,no=88,missing=88,gain=1554.58,cover=895007.0
                        87:[avg_u_st_topics64_cs-d_topics64<0.999999] yes=175,no=176,missing=176,gain=1045.08,cover=432493.0
                            175:leaf=-0.464526,cover=28607.5
                            176:leaf=-0.523946,cover=403885.0
                        88:[d_retrieve_vitemcf_sixhour<2.0] yes=177,no=178,missing=178,gain=1104.59,cover=462514.0
                            177:leaf=-0.509669,cover=67912.8
                            178:leaf=-0.551153,cover=394602.0
                    44:[avg_u_vdemand_ck-d_demand<0.613999] yes=89,no=90,missing=90,gain=1640.97,cover=459262.0
                        89:[d_retrieve_vals<2.0] yes=179,no=180,missing=180,gain=466.73,cover=277196.0
                            179:leaf=-0.496121,cover=73928.5
                            180:leaf=-0.524046,cover=203268.0
                        90:[d_retrieve_vals<2.0] yes=181,no=182,missing=182,gain=802.415,cover=182065.0
                            181:leaf=-0.451539,cover=60241.0
                            182:leaf=-0.493933,cover=121824.0
                22:[avg_u_vdemand_ck-d_demand<0.911999] yes=45,no=46,missing=46,gain=8148.85,cover=558425.0
                    45:[d_retrieve_vals<2.0] yes=91,no=92,missing=92,gain=1913.29,cover=407464.0
                        91:[uvd_ctravg<0.485999] yes=183,no=184,missing=183,gain=571.002,cover=92444.5
                            183:leaf=-0.46587,cover=58780.0
                            184:leaf=-0.416777,cover=33664.5
                        92:[avg_u_vdemand_ck-d_demand<0.561999] yes=185,no=186,missing=186,gain=939.473,cover=315019.0
                            185:leaf=-0.512764,cover=164944.0
                            186:leaf=-0.479913,cover=150075.0
                    46:[d_retrieve_vals<2.0] yes=93,no=94,missing=94,gain=2354.82,cover=150962.0
                        93:[uvd_ctravg<0.681999] yes=187,no=188,missing=187,gain=945.771,cover=53481.5
                            187:leaf=-0.383481,cover=34402.5
                            188:leaf=-0.300142,cover=19079.0
                        94:[d_retrieve_vitemcf_sixhour<2.0] yes=189,no=190,missing=190,gain=944.358,cover=97480.2
                            189:leaf=-0.392837,cover=35240.8
                            190:leaf=-0.454363,cover=62239.5
    2:[sud_ctravg<0.251999] yes=5,no=6,missing=5,gain=291429.0,cover=6776270.0
        5:[u_topics128_ck_127<0.224999] yes=11,no=12,missing=11,gain=47625.9,cover=4334130.0
            11:[avg_u_vdemand_ck-d_demand<0.493999] yes=23,no=24,missing=23,gain=23261.9,cover=2425710.0
                23:[d_stats_ectr<0.084999] yes=47,no=48,missing=48,gain=12270.7,cover=1932840.0
                    47:[sim_u_clk6h_topic256-d_topics256<2.0] yes=95,no=96,missing=96,gain=4152.97,cover=1180850.0
                        95:[uvd_ctravg<0.197999] yes=191,no=192,missing=191,gain=608.67,cover=204616.0
                            191:leaf=-0.515703,cover=132491.0
                            192:leaf=-0.481375,cover=72125.5
                        96:[uvd_ctravg<0.135999] yes=193,no=194,missing=193,gain=2776.52,cover=976237.0
                            193:leaf=-0.564176,cover=568844.0
                            194:leaf=-0.531712,cover=407393.0
                    48:[max_u_st_ls_ct_ck-d_ct<0.092999] yes=97,no=98,missing=97,gain=6172.5,cover=751991.0
                        97:[d_retrieve_als<2.0] yes=195,no=196,missing=196,gain=3393.27,cover=636830.0
                            195:leaf=-0.462172,cover=132049.0
                            196:leaf=-0.516211,cover=504781.0
                        98:[d_stats_rawctr<0.181999] yes=197,no=198,missing=198,gain=1160.63,cover=115161.0
                            197:leaf=-0.447751,cover=84301.2
                            198:leaf=-0.379693,cover=30860.0
                24:[avg_u_vdemand_ck-d_demand<0.911999] yes=49,no=50,missing=50,gain=3215.95,cover=492865.0
                    49:[sim_u_clk6h_tag-d_tags<0.045999] yes=99,no=100,missing=99,gain=1039.55,cover=320284.0
                        99:[sim_u_vdemand_ck-d_demand<0.830999] yes=199,no=200,missing=200,gain=487.136,cover=283520.0
                            199:leaf=-0.464731,cover=177881.0
                            200:leaf=-0.490517,cover=105640.0
                        100:[sim_u_vtopics128_cs-d_topics128<-1e-06] yes=201,no=202,missing=201,gain=314.629,cover=36764.0
                            201:leaf=-0.545817,cover=1732.0
                            202:leaf=-0.414458,cover=35032.0
                    50:[max_u_vkeywords_ck-d_titlekws<0.600999] yes=101,no=102,missing=101,gain=1428.22,cover=172581.0
                        101:[sim_u_vdemand_ck-d_demand<0.693999] yes=203,no=204,missing=204,gain=414.699,cover=104681.0
                            203:leaf=-0.416382,cover=42347.8
                            204:leaf=-0.454953,cover=62333.2
                        102:[sim_u_vtopics64_ck-d_topics64<0.316999] yes=205,no=206,missing=206,gain=515.019,cover=67899.8
                            205:leaf=-0.37204,cover=57063.0
                            206:leaf=-0.443492,cover=10836.8
            12:[u_demand_ck_NEWS<0.704999] yes=25,no=26,missing=25,gain=15052.1,cover=1908420.0
                25:[d_stats_ectr<0.093999] yes=51,no=52,missing=52,gain=4297.18,cover=1053730.0
                    51:[avg_u_vdemand_ck-d_demand<0.493999] yes=103,no=104,missing=103,gain=3339.58,cover=534901.0
                        103:[newsType<14.0] yes=207,no=208,missing=208,gain=1870.91,cover=407314.0
                            207:leaf=-0.486633,cover=267320.0
                            208:leaf=-0.529474,cover=139994.0
                        104:[uvd_ctravg<0.607999] yes=209,no=210,missing=209,gain=647.465,cover=127587.0
                            209:leaf=-0.457477,cover=97995.5
                            210:leaf=-0.406757,cover=29591.8
                    52:[sud_ctravg<0.102999] yes=105,no=106,missing=106,gain=2024.1,cover=518830.0
                        105:[uvd_ctravg<0.519999] yes=211,no=212,missing=211,gain=298.02,cover=125913.0
                            211:leaf=-0.48636,cover=119160.0
                            212:leaf=-0.421299,cover=6752.75
                        106:[ud_ctrmin<0.113999] yes=213,no=214,missing=213,gain=1519.08,cover=392917.0
                            213:leaf=-0.461181,cover=164120.0
                            214:leaf=-0.423331,cover=228797.0
                26:[d_retrieve_als<2.0] yes=53,no=54,missing=54,gain=9925.98,cover=854690.0
                    53:[u_topics128_ck_127<1.058] yes=107,no=108,missing=108,gain=2752.67,cover=188106.0
                        107:[d_stats_ectr<0.125999] yes=215,no=216,missing=216,gain=746.817,cover=132969.0
                            215:leaf=-0.39548,cover=83431.8
                            216:leaf=-0.348929,cover=49537.5
                        108:[sim_u_st_topics64_ck-d_topics64<2.0] yes=217,no=218,missing=218,gain=911.726,cover=55136.8
                            217:leaf=-0.338149,cover=26749.8
                            218:leaf=-0.26092,cover=28387.0
                    54:[avg_u_vdemand_ck-d_demand<0.755999] yes=109,no=110,missing=109,gain=5134.75,cover=666584.0
                        109:[newsType<14.0] yes=219,no=220,missing=220,gain=7426.63,cover=504860.0
                            219:leaf=-0.422908,cover=344718.0
                            220:leaf=-0.501104,cover=160142.0
                        110:[sim_u_vtopics64_ck-d_topics64<0.999999] yes=221,no=222,missing=222,gain=749.552,cover=161723.0
                            221:leaf=-0.383107,cover=157933.0
                            222:leaf=-0.518248,cover=3789.75
        6:[u_demand_ck_LEISURE<0.921999] yes=13,no=14,missing=14,gain=98175.4,cover=2442140.0
            13:[d_stats_ectr<0.111999] yes=27,no=28,missing=28,gain=21126.2,cover=1747360.0
                27:[ud_ctrmin<0.113999] yes=55,no=56,missing=55,gain=6649.97,cover=939560.0
                    55:[sim_u_clk6h_topic256-d_topics256<0.045999] yes=111,no=112,missing=111,gain=1746.11,cover=393058.0
                        111:[d_retrieve_als<2.0] yes=223,no=224,missing=224,gain=1046.34,cover=188152.0
                            223:leaf=-0.417454,cover=28699.8
                            224:leaf=-0.47975,cover=159452.0
                        112:[d_stats_clickNum<53.0] yes=225,no=226,missing=226,gain=825.741,cover=204906.0
                            225:leaf=-0.496245,cover=15763.0
                            226:leaf=-0.424691,cover=189143.0
                    56:[newsType<14.0] yes=113,no=114,missing=114,gain=3855.46,cover=546502.0
                        113:[sulsd_ctravg<0.270999] yes=227,no=228,missing=227,gain=2343.79,cover=461465.0
                            227:leaf=-0.423478,cover=119897.0
                            228:leaf=-0.374707,cover=341568.0
                        114:[avg_u_vtopics64_ck-d_topics64<0.122999] yes=229,no=230,missing=229,gain=1970.19,cover=85036.8
                            229:leaf=-0.510922,cover=35465.8
                            230:leaf=-0.418258,cover=49571.0
                28:[max_u_clk6h_tag-d_tags<5.799] yes=57,no=58,missing=57,gain=11358.6,cover=807800.0
                    57:[ud_ctrmin<0.197999] yes=115,no=116,missing=115,gain=8662.2,cover=734918.0
                        115:[ud_ctravg<0.208999] yes=231,no=232,missing=231,gain=2433.75,cover=396188.0
                            231:leaf=-0.432496,cover=110867.0
                            232:leaf=-0.3801,cover=285321.0
                        116:[min_u_st_ls_tags_cs-d_tags<0.061999] yes=233,no=234,missing=234,gain=2349.06,cover=338729.0
                            233:leaf=-0.29413,cover=113134.0
                            234:leaf=-0.347114,cover=225595.0
                    58:[max_u_st_ls_ct_ck-d_ct<0.271999] yes=117,no=118,missing=117,gain=2795.12,cover=72883.0
                        117:[sulsd_ctravg<0.194999] yes=235,no=236,missing=235,gain=483.216,cover=24745.2
                            235:leaf=-0.384151,cover=7819.75
                            236:leaf=-0.293876,cover=16925.5
                        118:[min_u_keywords_cs-d_contentkws<0.642999] yes=237,no=238,missing=238,gain=1565.32,cover=48137.8
                            237:leaf=-0.0641512,cover=6730.75
                            238:leaf=-0.220153,cover=41407.0
            14:[d_stats_rawctr<0.157999] yes=29,no=30,missing=30,gain=45110.5,cover=694784.0
                29:[newsType<14.0] yes=59,no=60,missing=60,gain=12810.9,cover=492591.0
                    59:[sud_ctravg<0.627999] yes=119,no=120,missing=120,gain=9276.16,cover=400651.0
                        119:[u_topics128_ck_107<0.288999] yes=239,no=240,missing=239,gain=3452.13,cover=276070.0
                            239:leaf=-0.366929,cover=75339.0
                            240:leaf=-0.291606,cover=200731.0
                        120:[ud_ctravg<0.518999] yes=241,no=242,missing=241,gain=4670.69,cover=124581.0
                            241:leaf=-0.262242,cover=73158.2
                            242:leaf=-0.144251,cover=51422.8
                    60:[uvd_ctravg<0.403999] yes=121,no=122,missing=121,gain=4027.48,cover=91940.5
                        121:[uvd_ctravg<0.219999] yes=243,no=244,missing=243,gain=427.626,cover=30113.0
                            243:leaf=-0.532239,cover=14750.8
                            244:leaf=-0.460498,cover=15362.2
                        122:[uvd_ctravg<0.607999] yes=245,no=246,missing=246,gain=921.065,cover=61827.5
                            245:leaf=-0.395298,cover=33711.0
                            246:leaf=-0.321707,cover=28116.5
                30:[max_u_clk5m_tag-d_tags<1.341] yes=61,no=62,missing=61,gain=15920.3,cover=202193.0
                    61:[ud_ctrmin<0.388999] yes=123,no=124,missing=123,gain=5977.36,cover=149340.0
                        123:[max_u_st_ls_ct_ck-d_ct<0.171999] yes=247,no=248,missing=247,gain=1676.14,cover=88713.0
                            247:leaf=-0.283183,cover=38496.0
                            248:leaf=-0.199966,cover=50217.0
                        124:[u_taste_cs_sick<-1.884] yes=249,no=250,missing=250,gain=2532.76,cover=60627.0
                            249:leaf=0.0105872,cover=11843.5
                            250:leaf=-0.14407,cover=48783.5
                    62:[u_topics128_ck_69<0.684999] yes=125,no=126,missing=125,gain=4742.18,cover=52853.0
                        125:[max_u_st_ls_ct_ck-d_ct<0.408999] yes=251,no=252,missing=251,gain=1298.86,cover=29765.8
                            251:leaf=-0.161006,cover=10169.8
                            252:leaf=-0.0288718,cover=19596.0
                        126:[avg_u_st_sct_ck-d_sct<0.499999] yes=253,no=254,missing=253,gain=1431.58,cover=23087.2
                            253:leaf=-0.0128095,cover=6451.25
                            254:leaf=0.15367,cover=16636.0
```

### 3.2 模型解析代码GBDTPredictor.py

```python
#-*- coding: UTF-8 -*-
import sys
import math
import codecs
reload(sys)
sys.setdefaultencoding("utf-8")
# 判断是否安装了xgboost和scipy
try:
    import xgboost as xgb
    import scipy.sparse.coo as coo
    have_xgboost_and_scipy = True
except:
    have_xgboost_and_scipy = False


def to_DMatrix(data, f_map):
    values = []
    rows = []
    cols = []

    for row, feature in enumerate(data):
        for fname, value in feature.items():
            if fname not in f_map:
                continue

            values.append(value)
            rows.append(row)
            cols.append(f_map[fname])

    arr = coo.coo_matrix((values, (rows, cols)), shape=(len(data), len(f_map)))
    return xgb.DMatrix(arr)


class TreeNode:
    def __init__(self,
                 is_leaf,
                 fname=None,
                 fvalue=None,
                 yesnode=None,
                 nonode=None,
                 missingnode=None,
                 gain=None,
                 cover=None,
                 weight=None):
        self.is_leaf = is_leaf
        self.fname = fname
        self.fvalue = fvalue
        self.yesnode = yesnode
        self.nonode = nonode
        self.missingnode = missingnode
        self.gain = gain
        self.cover = cover
        self.weight = weight


class GBDTPredictor:
    def __init__(
            self,
            dump_file,  # xgboost.Booster()的dump文件。该文件应当采用utf-8编码
            feature_list=None  # 特征名称列表
    ):
        self._load(dump_file, feature_list)

    # 解析模型文件
    def _load(self, dump_file, feature_list=None):
        self.forest = []

        for line in codecs.open(dump_file, encoding='UTF-8').readlines():
            # if line.strip().startswith('booster'):
            #     itree = int(line.split('[', 1)[1].split(']', 1)[0])
            #     continue

            if not line.strip()[0].isdigit():
                continue

            inode, info = line.strip().split(':', 1)
            inode = int(inode)

            if inode == 0:
                self.forest.append([])

            if info.startswith('['):  # 非叶子节点
                fname = info.split('[', 1)[1].split('<', 1)[0]
                fvalue = float(info.split('<', 1)[1].split(']', 1)[0])
                yesnode = int(info.split('yes=', 1)[1].split(',', 1)[0])
                nonode = int(info.split('no=', 1)[1].split(',', 1)[0])
                missingnode = int(
                    info.split('missing=', 1)[1].split(',', 1)[0])
                if info.find('gain=') >= 0:
                    gain = float(info.split('gain=', 1)[1].split(',', 1)[0])
                else:
                    gain = None
                if info.find('cover=') >= 0:
                    cover = float(info.split('cover=', 1)[1].split(',', 1)[0])
                else:
                    cover = None

                node = TreeNode(
                    False,
                    fname=fname,
                    fvalue=fvalue,
                    yesnode=yesnode,
                    nonode=nonode,
                    missingnode=missingnode,
                    gain=gain,
                    cover=cover)

            elif info.startswith('leaf'):  # 叶子节点
                weight = float(info.split('leaf=', 1)[1].split(',', 1)[0])
                if info.find('cover=') >= 0:
                    cover = float(info.split('cover=', 1)[1].split(',', 1)[0])
                else:
                    cover = None

                node = TreeNode(True, weight=weight, cover=cover)
            else:
                continue

            tree = self.forest[-1]
            tree.extend([None] * (inode + 1 - len(tree)))
            tree[inode] = node

        # 如果模型文件中只有特征序号而没有特征名，可以从列表中获取特征名称
        if feature_list:

            # 此循环用以判断各节点的fname是否都是f***的形式，其中***为数字
            # 如果不是该形式，则无法替换特征名
            for tree in self.forest:
                for node in tree:
                    fname = node.fname
                    if node.is_leaf:
                        continue
                    if fname[0] != 'f':
                        return
                    try:
                        i = int(fname[1:])
                    except:
                        return
                    if i > len(feature_list):
                        return

            # 替换特征名称
            for tree in self.forest:
                for node in tree:
                    if node.is_leaf:
                        continue
                    i = int(node.fname[1:])
                    node.fname = feature_list[i]

    def predict(
            self,
            features,  # map类型的特征。其中的特征名必须为Unicode
            ntree_limit=0,  # 使用前N棵树进行预测
            pred_leaf=False  # 若为True，则输出各树的叶子节点序号
    ):
        ntree_limit = int(ntree_limit)
        if ntree_limit <= 0 or ntree_limit > len(self.forest):
            ntree_limit = len(self.forest)

        if pred_leaf:
            return self._predict_leaf(features, ntree_limit)
        else:
            return self._predict_score(features, ntree_limit)

    # 返回每个特征的重要度。有三类重要度的度量：
    #   weight: 特征在所有树中出现的总次数
    #   gain: 特征的gain的平均值
    #   cover: 特征的cover的平均值
    def get_score(self, importance_type='weight'):
        if importance_type == 'weight':
            return self._get_feature_weight()
        elif importance_type == 'gain':
            return self._get_feature_gain()
        elif importance_type == 'cover':
            return self._get_feature_cover()

    def _get_feature_weight(self):
        weight = {}
        for tree in self.forest:
            for node in tree:
                if node.is_leaf:
                    continue
                fname = node.fname
                weight.setdefault(fname, 0)
                weight[fname] += 1

        return weight

    def _get_feature_gain(self):
        gain = {}
        for tree in self.forest:
            for node in tree:
                if node.is_leaf:
                    continue
                if node.gain:
                    fname = node.fname
                    gain.setdefault(fname, [0, 0.0])
                    gain[fname][0] += 1
                    gain[fname][1] += node.gain

        for k, v in gain.items():
            gain[k] = v[1] / v[0]

        return gain

    def _get_feature_cover(self):
        cover = {}
        for tree in self.forest:
            for node in tree:
                if node.is_leaf:
                    continue
                if node.cover:
                    fname = node.fname
                    cover.setdefault(fname, [0, 0.0])
                    cover[fname][0] += 1
                    cover[fname][1] += node.cover

        for k, v in cover.items():
            cover[k] = v[1] / v[0]

        return cover

    def _predict_leaf(self, features, ntree_limit):

        line_result = []
        for itree in range(ntree_limit):
            ileaf = self._locate_leaf_in_tree(itree, features)
            line_result.append(ileaf)

        return line_result

    def _predict_score(self, features, ntree_limit):

        score = 0.0

        for itree in range(ntree_limit):
            ileaf = self._locate_leaf_in_tree(itree, features)
            score += self.forest[itree][ileaf].weight

            # uncomment for debugging
            # print '%s' % self.forest[itree][ileaf].weight

        return 1.0 / (1.0 + math.exp(-score))

    def _locate_leaf_in_tree(self, itree, features):
        tree = self.forest[itree]

        # uncomment for debugging
        # print '%s\t' % itree,

        inode = 0
        while True:
            # print '%s\t' % inode,
            node = tree[inode]
            if node.is_leaf:
                return inode

            fname = node.fname
            if fname not in features:
                inode = node.missingnode
            elif float(features[fname]) < node.fvalue:
                inode = node.yesnode
            else:
                inode = node.nonode

    def dump_model(self, dump_file):
        fout = codecs.open(dump_file, 'w', encoding='utf-8')
        for itree, tree in enumerate(self.forest):
            fout.write('booster[%s]:\n' % itree)

            GBDTPredictor._print_subtree(fout, tree, 0, 0)

    @staticmethod
    def _print_subtree(fout, tree, inode, indent):
        node = tree[inode]

        if node.is_leaf:
            out_str = '%s:leaf=%s' % (inode, node.weight)
            if node.cover:
                out_str += ',cover=%s' % node.cover
        else:
            out_str = '%s:[%s<%s] yes=%s,no=%s,missing=%s' % (inode,
                                                              node.fname,
                                                              node.fvalue,
                                                              node.yesnode,
                                                              node.nonode,
                                                              node.missingnode)
            if node.gain:
                out_str += ',gain=%s' % node.gain
            if node.cover:
                out_str += ',cover=%s' % node.cover

        indent_str = ''.join(['    '] * indent)
        fout.write(indent_str + out_str + '\n')

        if node.is_leaf:
            return
        else:
            GBDTPredictor._print_subtree(fout, tree, node.yesnode, indent + 1)
            GBDTPredictor._print_subtree(fout, tree, node.nonode, indent + 1)

```

### 3.3 调用debug代码

```python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/26 5:01 下午
# @Author  : zhuwei
# @FileName: debugGBDT.py
# @Software: PyCharm
# @Blog    ：http://blog.csdn.net/u010105243/article/

import GBDTPredictor
import codecs
import math

gbdt_modelfile = "/Users/zenmen/Documents/fm_dir_gy/model_xgboost_2021_01_21_03"
#gbdt_pred = GBDTPredictor.GBDTPredictor(gbdt_modelfile)

def _load(dump_file, feature_list=None):
    forest = []
    debug_index = 0
    for line in codecs.open(dump_file, encoding='UTF-8').readlines():
        # if line.strip().startswith('booster'):
        #     itree = int(line.split('[', 1)[1].split(']', 1)[0])
        #     continue
        if not line.strip()[0].isdigit():
            continue
        inode, info = line.strip().split(':', 1)# 0 [u_demand_ck_LEISURE<0.071999] yes=1,no=2,missing=1,gain=879772.0,cover=29654000.0
        inode = int(inode)
        if inode == 0:# 表示一棵树的根结点
            forest.append([])
        if info.startswith('['):  # 非叶子节点
            fname = info.split('[', 1)[1].split('<', 1)[0]# u_demand_ck_LEISURE
            fvalue = float(info.split('<', 1)[1].split(']', 1)[0])# 0.071999
            yesnode = int(info.split('yes=', 1)[1].split(',', 1)[0])
            nonode = int(info.split('no=', 1)[1].split(',', 1)[0])
            missingnode = int(info.split('missing=', 1)[1].split(',', 1)[0])
            if info.find('gain=') >= 0:
                gain = float(info.split('gain=', 1)[1].split(',', 1)[0])
            else:
                gain = None
            if info.find('cover=') >= 0:
                cover = float(info.split('cover=', 1)[1].split(',', 1)[0])
            else:
                cover = None
            node = GBDTPredictor.TreeNode(
                False,
                fname=fname,
                fvalue=fvalue,
                yesnode=yesnode,
                nonode=nonode,
                missingnode=missingnode,
                gain=gain,
                cover=cover)
            #print("node.fname=", node.fname)
            #print(fname, fvalue, yesnode, nonode, missingnode, gain, cover)
        elif info.startswith('leaf'):  # 叶子节点
            weight = float(info.split('leaf=', 1)[1].split(',', 1)[0])  # 叶子节点的值
            if info.find('cover=') >= 0:
                cover = float(info.split('cover=', 1)[1].split(',', 1)[0])
            else:
                cover = None
            node = GBDTPredictor.TreeNode(True, weight=weight, cover=cover)
        else:
            continue
        tree = forest[-1] #修改了a，就影响了b；同理，修改了b就影响了a。
        tree.extend([None] * (inode + 1 - len(tree)))
        debug_index += 1
        # if debug_index >= 5:
        #     break
        tree[inode] = node # 每个节点存入tree中

    # 如果模型文件中只有特征序号而没有特征名，可以从列表中获取特征名称
    if feature_list:
            # 此循环用以判断各节点的fname是否都是f***的形式，其中***为数字
            # 如果不是该形式，则无法替换特征名
            for tree in forest: # 遍历每棵树
                for node in tree: # 每个树中的每个节点
                    fname = node.fname
                    if node.is_leaf:# 叶子节点
                        continue
                    if fname[0] != 'f':
                        return
                    try:
                        i = int(fname[1:])
                    except:
                        return
                    if i > len(feature_list):
                        return

            # 替换特征名称
            for tree in forest:
                for node in tree:
                    if node.is_leaf:
                        continue
                    i = int(node.fname[1:])
                    node.fname = feature_list[i]
    return forest
    #print(len(forest), forest)

def predict(
    forests,
    features,  # map类型的特征。其中的特征名必须为Unicode
    ntree_limit=0,  # 使用前N棵树进行预测
    pred_leaf=False  # 若为True，则输出各树的叶子节点序号
    ):
    ntree_limit = int(ntree_limit)
    if ntree_limit <= 0 or ntree_limit > len(forests):
            ntree_limit = len(forests)   # 树的棵数300

    if pred_leaf: # True
        return _predict_leaf(features, ntree_limit)
    else:
        return _predict_score(features, ntree_limit)

def _locate_leaf_in_tree(forests, itree, features):
        tree = forests[itree]
        # uncomment for debugging
        # print '%s\t' % itree,
        inode = 0
        while True:
            # print '%s\t' % inode,
            node = tree[inode]
            if node.is_leaf: # 一直循环，直到叶子结点
                return inode
            fname = node.fname
            if fname not in features:
                inode = node.missingnode
            elif float(features[fname]) < node.fvalue:
                inode = node.yesnode
            else:
                inode = node.nonode

# 将叶子节点作为特征，进行离散化
def _predict_leaf(features, ntree_limit):
        line_result = []
        for itree in range(ntree_limit):# 遍历每棵树
            ileaf = _locate_leaf_in_tree(itree, features) # 返回叶子节点
            line_result.append(ileaf)
        return line_result
# 计算预测分数
def _predict_score(features, ntree_limit, forests):
        score = 0.0
        for itree in range(ntree_limit):# 遍历每棵树
            ileaf = _locate_leaf_in_tree(itree, features)
            score += forests[itree][ileaf].weight
            # uncomment for debugging
            # print '%s' % self.forest[itree][ileaf].weight
        return 1.0 / (1.0 + math.exp(-score))

if __name__ == "__main__":
    forests = _load(gbdt_modelfile, "123")
    features = None
    pred_leaf = predict(forests, features, pred_leaf=True)
    leaf_feat = {}  # GBDT高阶特征
    for i, leaf_no in enumerate(pred_leaf):
        leaf_feat['tree_%d_%d' % (i, leaf_no)] = 1
```

### 3.4 将模型文件可视化为树结构





