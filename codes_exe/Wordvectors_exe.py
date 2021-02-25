from sklearn.feature_extraction.text import CountVectorizer
corpus = [
  'I like this course.',
  'I like this game.',
  'I like this course, but I also like that game',
]
# 构建countvectorizer object
vectorizer = CountVectorizer()
# 得到每个文档的count向量
X = vectorizer.fit_transform(corpus)
# 打印词典
print("vocabulary={}".format(vectorizer.get_feature_names())) # ['also', 'but', 'course', 'game', 'like', 'that', 'this']
# 打印每个文档的向量
print("vectors={}".format(X.toarray()))

"""
print_res:
vocabulary=['also', 'but', 'course', 'game', 'like', 'that', 'this']
vectors=[[0 0 1 0 1 0 1]
 [0 0 0 1 1 0 1]
 [1 1 1 1 2 1 1]]
"""
