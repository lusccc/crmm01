import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

# 加载数据集
url = 'https://raw.githubusercontent.com/Agewerc/ML-Finance/master/data/corporate_rating.csv'
df = pd.read_csv(url)

# 提取特征和标签
X = df.drop(columns=['Name', 'Symbol', 'Rating Agency Name', 'Date', 'Sector', 'Rating'])
y = df['Rating']

# 提取非数值型特征，并对其进行独热编码
cat_cols = ['Industry']
ct = make_column_transformer((OneHotEncoder(), cat_cols))
X_cat = ct.fit_transform(df[cat_cols])

# 将独热编码后的特征和数值型特征合并
X = pd.concat([pd.DataFrame(X_cat.toarray()), X], axis=1)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林分类器
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 在测试集上进行预测，并计算准确率
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('准确率：', acc)
