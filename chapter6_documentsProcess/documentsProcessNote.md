# 一、文本分析整体流程

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210209163942526.png" alt="image-20210209163942526" style="zoom:50%;" />

# 二、分词

## 2.1 常用中文分词工具

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210209164437724.png" alt="image-20210209164437724" style="zoom:50%;" />

举例

```python
# encoding=utf-8
import jieba
# 基于jieba的分词, 结巴词库不包含"贪心学院"关键词
seg_list = jieba.cut("贪心学院专注于人工智能教育", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))
jieba.add_word("贪心学院") # 加入关键词
seg_list = jieba.cut("贪心学院专注于人工智能教育", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))
```

结果:

```shell
Default Mode: 贪心/ 学院/ 专注/ 于/ 人工智能/ 教育
Default Mode: 贪心学院/ 专注/ 于/ 人工智能/ 教育
```

## 2.2 最大匹配算法

词典:{"我们"，“经常”，“有”，“有意见”，“意见”， “分歧”}；

句子:"我们经常有意见分歧"，h=5;

### 1、前向最大匹配算法

step1:
$$
我们经常有(没有在词典里);==>我们经常(没有在词典里);==>我们经(没有在词典里);==>我们(在词典里)
$$
因此"我们"这个词被分出来了；

step2:
$$
经常有意见(没有在词典里);==>经常有意(没有在词典里);==>经常有(没有在词典里);==>经常(在词典里);
$$
因此“经常”这个词被分出来了；

step3:
$$
有意见分歧(没有在词典里);==>有意见分(没有在词典里);==>有意见(在词典里);
$$
因此“有意见”这个词被分出来了；

step4:

"分歧"这个词被分出来了；

最终结果:"我们|经常|有意见|分歧"；

### 2、后向最大匹配算法

step1:
$$
有意见分歧(没有在词典里);==>意见分歧(没有在词典里);==>见分歧(没有在词典里);==>分歧(在词典里)
$$
因此“分歧”这个词被分出来了；

step2:
$$
经常有意见(没有在词典里);==>常有意见(没有在词典里);==>有意见(在词典里);
$$
因此“有意见”这个词被分出来了；

step3:
$$
我们经常((没有在词典里);==>们经常((没有在词典里);==>经常((在词典里);
$$
因此“经常”这个词被分出来了；

Step4:

"我们"这个词被分出来了；

最终结果:"我们|经常|有意见|分歧"；

### 3、缺点

没有考虑单词之间的语义和语义关系；

## 2.3 考虑语义的方法

输入==>生成所有可能的分割==>选择其中最好的；

句子:"我们经常有意见分歧";
$$
P_{LM}(我们,经常,有意见,分歧)=p1;\\
P_{LM}(我们,经常,有,意见,分歧)=p2;\\
P_{LM}(我们,经常有,意见,分歧)=p3;\\
................\\
p2最大，因此(我们,经常,有,意见,分歧)是最好的分词；
假设LM是unigram语言模型；\\
P_{LM}(我们,经常,有意见,分歧)=P_{LM}(我们)*P_{LM}(经常)*P_{LM}(有意见)*P_{LM}(分歧)\\
为了防止溢出:P_{LM}(我们,经常,有意见,分歧)=logP_{LM}(我们)+logP_{LM}(经常)+logP_{LM}(有意见)+logP_{LM}(分歧)
$$

## 2.4 维特比算法

例子:经常有意见分歧

词典:[经常，经，有，有意见，意见，分歧，见，意，见分歧，分]；

概率:[0.1, 0.05, 0.1, 0.1, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1];

-log(x):[2.3, 3, 2.3, 2.3, 1.6, 1.6, 3, 3, 3, 2.3];

解决方案:

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210210083540392.png" alt="image-20210210083540392" style="zoom:50%;" />

计算最优解(动态规划):

1、状态定义 :f(i)表示从节点1到当前节点的最短路径 ；本题是求f(8);

2、状态计算:f[i] = min(f[x]+path,f[i]);

例如f[8] = min(f[7]+1000, f[6]+1.6, f[5]+3);

## 2.5 分词总结

### 1、基于规则匹配的方法；

### 2、基于概率统计方法（LM，HMM，CRF）；

### 3、分词可以认为是已经解决的问题；

# 三、停用词已经词的过滤

通用先把停用词，出现频率很低的词过滤掉；

```python
# 方法1: 自己建立一个停用词词典
stop_words = ["the", "an", "is", "there"]
# 在使用时: 假设 word_list包含了文本里的单词
word_list = ["we", "are", "the", "students"]
filtered_words = [word for word in word_list if word not in stop_words]
print (filtered_words)
# 方法2:直接利用别人已经构建好的停用词库
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")
```

## 3.1 词的标准化Stemming,Lemmazation

go, went,going;==>go

Fly,flies;==>flies

deny,denied,denying;==>deni

fast,faster,fastest;==>fast

使用stemming标准化后的单词未必在词库里，而lemmazation标准化的词都在词库里；

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210210092137570.png" alt="image-20210210092137570" style="zoom:50%;" />

测试代码：

```python
from nltk.stem.porter import *
stemmer = PorterStemmer()
test_strs = ['caresses', 'flies', 'dies', 'mules', 'denied',
    'died', 'agreed', 'owned', 'humbled', 'sized',
    'meeting', 'stating', 'siezing', 'itemization',
    'sensational', 'traditional', 'reference', 'colonizer',
    'plotted']
singles = [stemmer.stem(word) for word in test_strs]
print(' '.join(singles)) # doctest: +NORMALIZE_WHITESPACE
结果：
caress fli die mule deni die agre own humbl size meet state siez item sensat tradit refer colon plot
```

# 四、拼写纠错

单词错误，语法错误；

## 4.1 编辑距离

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210210093111213.png" alt="image-20210210093111213" style="zoom:50%;" />

代码:

```python

# 基于动态规划的解法
def edit_dist(s, t):
 # str1, str2为两个输入字符串
  # m,n分别字符串str1和str2的长度
  m, n = len(s), len(t)
 
  # 构建二位数组来存储子问题(sub-problem)的答案
  dp = [[0 for x in range(n+1)] for x in range(m+1)]
  for i in range(n+1):dp[i][0] = i
  for j in range(m+1):dp[0][j] = j
  # 利用动态规划算法,填充数组
  for i in range(1, m+1):
    for j in range(1, n+1):
        dp[i][j] = min(dp[i][j-1]+1, dp[i-1][j]+1)
        if s[i-1] != t[j-1]:
            dp[i][j] = min(dp[i][j], dp[i-1][j-1]+1)
        else:
            dp[i][j] = min(dp[i][j], dp[i-1][j-1])
  return dp[m][n]
print ("edit_distance={}".format(edit_dist("horse", "ros")))
```

## 4.2 改进方法

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210210141251167.png" alt="image-20210210141251167" style="zoom:50%;" />

两种方法比较:
$$
1、用户输入-->从词典中选择编辑距离最小的(会遍历整个词典)-->返回;\\
2、用户输入-->生成编辑距离为1，2的字符串-->过滤-->返回;
$$
代码：

```python
def generate_edit_one(str):
  """
  给定一个字符串,生成编辑距离为1的字符串列表。
  """
  letters  = 'abcdefghijklmnopqrstuvwxyz'
  splits = [(str[:i], str[i:]) for i in range(len(str)+1)]
  inserts = [L + c + R for L, R in splits for c in letters]
  deletes = [L + R[1:] for L, R in splits if R]
  replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
 
  #return set(splits)
  return set(inserts + deletes + replaces)
def generate_edit_two(str):
  """
  给定一个字符串,生成编辑距离不大于2的字符串
  """
  return [e2 for e1 in generate_edit_one(str) for e2 in generate_edit_one(e1)]
# 测试
print (len(generate_edit_two("apple")))
print (len(generate_edit_two("apple")))
```

## 4.3 寻找最好的候选单词

<img src="/Users/zenmen/Projects/courses_ml_notebook/images/image-20210210143047427.png" alt="image-20210210143047427" style="zoom:50%;" />

给定一个字符串s,我们要找出最有可能成为正确的字符串c,也就是$c=argmac_{c\in candidates}p(c|s)$
$$
argmac_{c\in candidates}p(c|s)=argmax_{c\in candidates}\frac{p(c)p(s|c)}{p(s)}=argmax_{c\in candidates}p(c)p(s|c)
$$
对于拼写纠错，我们来做简单的总结:

- 第一步:找到拼写错误的单词
- 第二步:生成跟上述单词类似的其他单词，当作是候选集
- 第三步:根据单词在上下文中的统计信息来排序并选出最好的。

