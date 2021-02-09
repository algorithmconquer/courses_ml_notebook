## 员工离职率预测小案例
# 引入相应的工具包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
%matplotlib inline
from sklearn.model_selection import train_test_split
# 读取数据到pandas dataframe
df = pd.read_csv('/home/anaconda/data/Z_NLP/HR_comma_sep.csv', index_col=None)
# 检测是否有缺失数据
print (df.isnull().any(), "\n\n")
# 看看数据的样例吧
print (df.head(), "\n\n")
# 给定数据里的列名有些不太清楚,咱们改改吧!
df = df.rename(columns={'satisfaction_level': 'satisfaction',
            'last_evaluation': 'evaluation',
            'number_project': 'projectCount',
            'average_montly_hours': 'averageMonthlyHours',
            'time_spend_company': 'yearsAtCompany',
            'Work_accident': 'workAccident',
            'promotion_last_5years': 'promotion',
            'sales' : 'department',
            'left' : 'turnover'
            })

# 将预测标签‘是否离职’放在第一列,这是咱们的label!
front = df['turnover']
df.drop(labels=['turnover'], axis=1, inplace = True)
df.insert(0, 'turnover', front)
#df.head()
# 计算一下离职员工的百分比和没有离职的百分比。
turnover_rate = df.turnover.value_counts() / len(df)
print ("样本数据中,离职率为:%.2f\n\n" % turnover_rate[1])
# 最后显示一下统计数据看看吧!
print (df.describe(),"\n\n")
# 将string类型转换为整数类型,不然后面处理不了。
df["department"] = df["department"].astype('category').cat.codes
df["salary"] = df["salary"].astype('category').cat.codes
# 设置特征值和标签。 X 存放特征, y存放标签
target_name = 'turnover'
X = df.drop('turnover', axis=1)
y = df[target_name]
# 将数据分为训练和测试数据集
# 注意参数 stratify = y 意味着在产生训练和测试数据中, 离职的员工的百分比等于原来总的数据中的离职的员工的百分比
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
# 准备工作就绪,接下来就训练模型时间到了!
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# 决训练一下决策树
dtree = tree.DecisionTreeClassifier(
  criterion='entropy',
  #max_depth=3, # 定义树的深度, 可以用来防止过拟合
  min_weight_fraction_leaf=0.01 # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
  )
dtree = dtree.fit(X_train,y_train)
print ("\n\n ---决策树---")
print(classification_report(y_test, dtree.predict(X_test)))
# 随机森林
rf = RandomForestClassifier(
  criterion='entropy',
  n_estimators=1000,
  max_depth=None, # 定义树的深度, 可以用来防止过拟合
  min_samples_split=10, # 定义至少多少个样本的情况下才继续分叉
  #min_weight_fraction_leaf=0.02 # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
  )
rf.fit(X_train, y_train)
print ("\n\n ---随机森林---")
print(classification_report(y_test, rf.predict(X_test)))

