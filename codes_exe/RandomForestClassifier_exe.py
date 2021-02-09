
# 导入数字识别数据集,这个数据集已经集成在了sklearn里
from sklearn.datasets import load_digits
# 导入随机森林分类器
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# 导入数据
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X,
              y, test_size=0.2, random_state=42)
# 创建随机森林,参数可以适当修改一下。
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
clf = RandomForestClassifier(n_estimators=400, criterion='entropy',
              max_depth=5, min_samples_split=3, max_features='sqrt',random_state=0)
clf.fit(X_train, y_train)
print ("训练集上的准确率为:%.2f, 测试数据上的准确率为:%.2f"
   % (clf.score(X_train, y_train), clf.score(X_test, y_test)))

