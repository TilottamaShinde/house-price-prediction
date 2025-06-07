import pandas as pd
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split

train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Load Dataset
url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
df = pd.read_csv(url)

# Select features and target
X = df[['rm', 'lstat', 'ptratio']]
y = df['medv']

# Split dataset into training and test sets
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Build and train model
model =  LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
score = r2_score(y_test, y_pred)
print(f"R2 score : {score: .2f}")































