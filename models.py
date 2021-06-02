import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from data.acs_load import acs_load
import matplotlib.pyplot as plt
import contextily as ctx


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
    city = "Los Angeles"
    geometries, X, y = load_data(city)

    # Model objects
    svr = SVR(C=20, gamma=0.0001)

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

    # Fit SVM
    svr.fit(X_train, y_train)
    print(svr.score(X_train, y_train))
    print(svr.score(X_test, y_test))

    # Get predictions
    comparison = geometries.loc[test_index, :]
    comparison["predictions"] = svr.predict(X_test)
    comparison = geometries[["geometry"]].merge(comparison[["log_B19013001", "predictions"]],
                                                how="left",
                                                left_index=True,
                                                right_index=True)
    # Conver geography to Web Mercator (EPSG 3857)
    comparison = comparison.to_crs(epsg=3857)
    # Plot predictions vs actual
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, dpi=200,
                             figsize=(15, 10))
    comparison.plot(column="log_B19013001", ax=axes[0], cmap="viridis",
                    missing_kwds={"color": "lightgrey"}, alpha=0.7,
                    vmin=8.44, vmax=12.5)
    ctx.add_basemap(ax=axes[0], source=ctx.providers.CartoDB.Voyager)
    comparison.plot(column="predictions", ax=axes[1], cmap="viridis",
                    missing_kwds={"color": "lightgrey"}, alpha=0.7,
                    vmin=8.44, vmax=12.5)
    ctx.add_basemap(ax=axes[1], source=ctx.providers.CartoDB.Voyager)
    cols = axes[0].collections[0]
    colbar = fig.colorbar(cols, ax=axes, shrink=0.6)
    fig.patch.set_visible(False)
    axes[0].axis("off")
    axes[1].axis("off")
    axes[0].set_title("True Income")
    axes[1].set_title("""Predicted Income $(R^2 = 0.48)$""")
    # plt.savefig("./model visualizations/predictions_map.png",
    #             bbox_inches="tight")
    plt.show()

    # Scatter test set predictions
    plt.figure(dpi=200)
    plt.scatter(comparison["predictions"], comparison["log_B19013001"],
                s=2, color="#4A89F3")
    plt.xlabel("Predicted Log Income")
    plt.ylabel("True Log Income")
    plt.xlim(9, 13)
    plt.ylim(8, 13)
    plt.axline(xy1=(0, 0), slope=1, linestyle="--", color="#EA4335")
    # plt.savefig("./model visualizations/predictions_scatter.png",
    #             bbox_inches="tight")
    plt.show()

    # Scatter train set predictions
    plt.figure(dpi=200)
    plt.scatter(svr.predict(X_train), y_train,
                s=2, color="#4A89F3")
    plt.xlabel("Predicted Log Income")
    plt.ylabel("True Log Income")
    plt.xlim(9, 13)
    plt.ylim(8, 13)
    plt.axline(xy1=(0, 0), slope=1, linestyle="--", color="#EA4335")
    # plt.savefig("./model visualizations/predictions_scatter.png",
    #             bbox_inches="tight")
    plt.show()

    # Test a model on New York
    geometries, X, y = load_data(city="New York")
    X = X[:, :-1]
    X = scaler.fit_transform(X)
    print(svr.score(X, y))
    # Scatter New York predictions
    plt.figure(dpi=200)
    plt.scatter(svr.predict(X), y,
                s=2, color="#4A89F3")
    plt.xlabel("Predicted Log Income")
    plt.ylabel("True Log Income")
    plt.xlim(9, 13)
    plt.ylim(8, 13)
    plt.axline(xy1=(0, 0), slope=1, linestyle="--", color="#EA4335")
    # plt.savefig("./model visualizations/predictions_scatter.png",
    #             bbox_inches="tight")
    plt.show()
