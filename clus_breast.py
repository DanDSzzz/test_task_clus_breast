import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
type(cancer)
cancer.keys()
for k in cancer.keys():
  print(k)

print(type(cancer.data), type(cancer.target))
cancer.target_names
print(cancer.DESCR)
cancer_df = pd.DataFrame(cancer.data, columns = cancer.feature_names)


cancer_df['target'] = cancer.target

print(cancer_df.head())
unique, counts = np.unique(cancer.target, return_counts = True)
unique, counts
print(cancer_df.info())
print(cancer_df.describe().round(2))
print(cancer_df.isnull().sum())
cancer_df = pd.DataFrame(cancer.data, columns = cancer.feature_names)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_data = scaler.fit_transform(cancer_df)


type(scaled_data)
cancer_df_scaled = pd.DataFrame(scaled_data, columns = cancer.feature_names)


cancer_df_scaled['target'] = cancer.target


print(cancer_df_scaled.head(2))
print(cancer_df_scaled.describe().round(2))
data = cancer_df_scaled.groupby('target').mean().T

data.head(2)
data['diff'] = abs(data.iloc[:, 0] - data.iloc[:, 1])

data = data.sort_values(by = ['diff'], ascending = False)

data.head(10)

bins = 17

plt.figure(figsize = (10,6))


plt.hist(cancer_df_scaled.loc[cancer_df_scaled['target'] == 0, 'worst concave points'], bins, alpha = 0.5, label = 'Злокачественная')


plt.hist(cancer_df_scaled.loc[cancer_df_scaled['target'] == 1, 'worst concave points'], bins, alpha = 0.5, label = 'Доброкачественная')
plt.legend(loc = 'upper right')

plt.xlabel('worst concave points', fontsize = 16)
plt.ylabel('Количество наблюдений', fontsize = 16)
plt.title('Распределение worst concave points для двух типов опухолей', fontsize = 16)

features = list(data.index[:10])
print(features)
X = cancer_df_scaled[features]

y = cancer_df_scaled['target']
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 42)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix

model_matrix = confusion_matrix(y_test, y_pred, labels = [1,0])


model_matrix_df = pd.DataFrame(model_matrix)
print(model_matrix_df)
model_matrix_df = pd.DataFrame(model_matrix, columns = ['Прогноз добр.', 'Прогноз злок.'], index = ['Факт добр.', 'Факт злок.'])
print(model_matrix_df)
unique, counts = np.unique(y_pred, return_counts = True)
unique, counts
unique, counts = np.unique(y_test, return_counts = True)
unique, counts
round((61 + 104)/(61 + 104 + 2 + 4), 2)
from sklearn.metrics import accuracy_score

model_accuracy = accuracy_score(y_test, y_pred)
round(model_accuracy, 2)