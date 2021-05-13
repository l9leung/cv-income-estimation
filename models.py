import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# VGG16 extracted features
X = pd.read_csv("X_la_vgg16.csv", index_col=0)
X["geoid"] = X.index.str.split("_").str[0]
# Median household income
y = gpd.read_file(f"data/acs/Los Angeles/acs2019_5yr_B19013_15000US060372732001.geojson")
y = y[["geoid", "B19013001"]]
y["B19013001"] = np.log(y["B19013001"])  # Log transform

# Join
la = X.merge(y, on="geoid")
la = la.dropna()

X = np.array(la.drop(columns=["geoid", "B19013001"]).values)
y = np.array(la["B19013001"].values)
del la

# Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Fit models
param_grid = [{"C": [0.001, 0.1, 1, 10, 20], "kernel": ["linear"]},
              {"C": [0.001, 0.1, 1, 10, 20], "gamma": [0.1, 0.001, 0.0001], "kernel": ["rbf"]}]
reg = GridSearchCV(estimator=SVR(verbose=1), param_grid=param_grid, n_jobs=-1,
                   cv=5, verbose=4, return_train_score=True)
reg.fit(X, y)
reg.cv_results_
