import json
import pandas as pd
from data.acs_load import acs_load
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def load_data(city="Los Angeles", split=False):
    # VGG16 extracted features
    X = pd.read_csv(f"X_{city}_vgg16.csv", index_col=0)
    X["geoid"] = X.index.str.split("_").str[0]

    # Image coordinates
    with open(f"./data/street_view/{city}/coordinates.geojson") as file:
        points = json.load(file)
    coords = []
    for filename in points.keys():
        coords.append([f"{filename}.jpeg",
                       points[filename]["coordinates"][0],
                       points[filename]["coordinates"][1]])
    coords = pd.DataFrame(coords, columns=["filename",
                                           "longitude",
                                           "latitude"])

    # Median household income
    geometries = acs_load(cities=[city])[city]
    geometries = geometries[["geoid", "log_B19013001", "geometry"]]
    geometries = geometries.set_index("geoid")

    # Join with latitude-longitude
    X = coords.merge(X, left_on="filename", right_index=True)
    X.drop(columns="filename", inplace=True)

    # Aggregate by block group
    X = X.groupby("geoid").mean()

    # Join with geometry and income
    data = X.merge(geometries["log_B19013001"], left_index=True,
                   right_index=True)

    # Drop missing
    data = data.dropna()
    data["geoid"] = data.index

    X = data.drop(columns=["log_B19013001"]).to_numpy()
    y = data["log_B19013001"].to_numpy()

    return geometries, X, y


if __name__ == "__main__":
    # Load data
    city = "Los Angeles"
    geometries, X, y = load_data(city)
    X = X[:, 2:]  # Remove latitude-longitude

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                        random_state=412)
    train_index = X_train[:, -1]
    X_train = X_train[:, :-1]
    test_index = X_test[:, -1]
    X_test = X_test[:, :-1]

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model objects
    lasso = LassoCV(max_iter=5000, n_alphas=100, cv=3, verbose=True, n_jobs=-1)
    ridge = RidgeCV(alphas=np.geomspace(0.1, 1.2),cv=3)
    svr_lin = GridSearchCV(SVR(kernel="linear"),
                           param_grid={"C": [1, 0.1, 0.01, 0.001, 0.005]},
                           cv=3, verbose=4, n_jobs=-1)
    svr_rbf = GridSearchCV(SVR(kernel="rbf"),
                           param_grid={"C": [1, 0.1, 0.01, 0.001, 0.005],
                                       "gamma": ["scale", "auto"]},
                           cv=3, verbose=4, n_jobs=-1)
    tree = GridSearchCV(DecisionTreeRegressor(random_state=412),
                        param_grid={"max_depth": [None, 100, 500, 1000],
                                    "min_samples_leaf": [1, 5, 10, 100]},
                        cv=3, verbose=4, n_jobs=-1)
    rf = GridSearchCV(RandomForestRegressor(),
                      param_grid={"max_depth": [None, 100, 500, 1000],
                                  "min_samples_leaf": [1, 5, 10, 100]},
                      cv=3, verbose=4, n_jobs=-1)
    boosted_trees = GridSearchCV(GradientBoostingRegressor(),
                                 param_grid={"n_estimators": [10, 50, 100, 200],
                                             "max_depth": [1, 2, 3, 5, 10]},
                                 cv=3, verbose=4, n_jobs=-1)

    # Fit models
    lasso.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    svr_lin.fit(X_train, y_train)
    svr_rbf.fit(X_train, y_train)
    tree.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    boosted_trees.fit(X_train, y_train)

    svr_lin = svr_lin.best_estimator_
    svr_rbf = svr_rbf.best_estimator_
    tree = tree.best_estimator_
    rf = rf.best_estimator_
    boosted_trees = boosted_trees.best_estimator_

    # Evaluate models
    print(lasso.score(X_test, y_test))
    print(ridge.score(X_test, y_test))
    print(svr_lin.score(X_test, y_test))
    print(svr_rbf.score(X_test, y_test))
    print(tree.score(X_test, y_test))
    print(rf.score(X_test, y_test))
    print(boosted_trees.score(X_test, y_test))
