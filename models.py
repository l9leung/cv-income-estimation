import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

la = pd.read_csv("D:/la.csv", index_col=0)
la = pd.read_csv("D:/la_tester.csv", index_col=0)

X = np.array(la.drop(columns=["geoid", "log_B19013001"]).values)
y = np.array(la["log_B19013001"].values)

# Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Fit model
reg = SVR(kernel="rbf", C=0.5, verbose=True)
# Cross-validation
scores = cross_val_score(reg, X, y, cv=2)
print(scores)


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    shuffle=True,
                                                    random_state=123)
# Standardization using train set
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Fit model
reg = SVR(kernel="rbf", C=0.1, verbose=True)
reg.fit(X_train, y_train)
# Evaluate model
reg.score(X_test, y_test)
