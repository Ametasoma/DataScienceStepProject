import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = sns.load_dataset('titanic') #потрібний набір даних

print(df.head())
print(df.info())

df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]  #основні змінні моделі

df['family_size'] = df['sibsp'] + df['parch'] + 1

df['is_alone'] = (df['family_size'] == 1).astype(int)

#гістограмв вікових категорій
plt.figure(figsize=(6, 4))
sns.histplot(df['age'], kde=True, bins=30)
plt.title('Розподіл віку пасажирів')
plt.xlabel('Вік')
plt.ylabel('Кількість')

#коробкова діаграмма виживання за віком і статтю
plt.figure(figsize=(6, 4))
sns.boxplot(x='sex', y='age', hue='survived', data=df)
plt.title('Вік, стать і виживання')

#теплова карта кореляцій між числовими змінними
plt.figure(figsize=(8, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Кореляційна матриця')

plt.tight_layout()
plt.show()

#підготовка даних для моделі
X = df.drop('survived', axis=1)
y = df['survived']

#розділення на числові та категоріальні ознаки
numeric_features = ['age', 'fare', 'family_size']
categorical_features = ['pclass', 'sex', 'embarked', 'is_alone']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#об'єднання обробки
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

#тренувальні та тестові дані
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

#найкращі параметри
param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [5, 10, None]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Найкращі параметри:", grid_search.best_params_)

y_pred = best_model.predict(X_test)

print("Матриця помилок:")
print(confusion_matrix(y_test, y_pred))
print("\nЗвіт класифікації:")
print(classification_report(y_test, y_pred))