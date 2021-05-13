import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.optimizers import Adam
from data.acs_load import acs_load
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def build_model():
    model = Sequential()
    vgg16_model = VGG16(include_top=True, weights="imagenet")
    for layer in vgg16_model.layers[:-1]:
        model.add(layer)
    # for layer in model.layers:
    #     layer.trainable = False
    model.add(Dense(512, activation="relu", input_shape=(4096,)))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse",
                  metrics="mae")
    return model


def calc_R_2(y_hat, y_test):
    """Returns the coefficient of determination $R^2$ given predicted values
    y_hat and true values y_test."""
    SSR = ((y_test - y_hat)**2).sum()
    SST = ((y_test - y_test.mean())**2).sum()

    return 1 - (SSR/SST)


if __name__ == "__main__":
    # Image file names
    label_df = pd.DataFrame({"filenames": os.listdir("./data/street_view/Los Angeles/images")})
    label_df["geoid"] = label_df["filenames"].str.split("_", expand=True)[0]
    # Block group median income
    acs = acs_load(cities=["Los Angeles"])["Los Angeles"]
    acs = acs[["geoid", "log_B19013001"]]
    # Join file names with median incomes
    label_df = label_df.merge(acs, how="inner", on="geoid")
    del acs
    # Drop images with missing median incomes
    label_df = label_df.dropna()
    # Shuffle dataframe
    label_df = label_df.sample(frac=1, random_state=123).reset_index(drop=True)
    # Use 19000 of samples for training and validation, 5290 for testing
    train_label_df = label_df[:20000]
    test_label_df = label_df[20000:].copy()
    # Use 5000 of training samples for validation
    validation_label_df = train_label_df[15000:20000].copy()
    train_label_df = train_label_df[:15000]

    # Standardize log income using training set range
    scaler = StandardScaler().fit(train_label_df[["log_B19013001"]])
    train_label_df["log_B19013001"] = scaler.transform(train_label_df[["log_B19013001"]].astype("float32"))
    validation_label_df["log_B19013001"] = scaler.transform(validation_label_df[["log_B19013001"]].astype("float32"))
    test_label_df["log_B19013001"] = scaler.transform(test_label_df[["log_B19013001"]].astype("float32"))

    # Image data generators
    train_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
    validation_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
    batch_size = 100
    train_generator = train_datagen.flow_from_dataframe(dataframe=train_label_df,
                                                        directory="data/street_view/Los Angeles/images",
                                                        x_col="filenames",
                                                        y_col="log_B19013001",
                                                        target_size=(224, 224),
                                                        color_mode="rgb",
                                                        class_mode="raw",
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        seed=123)
    validation_generator = validation_datagen.flow_from_dataframe(dataframe=validation_label_df,
                                                                  directory="data/street_view/Los Angeles/images",
                                                                  x_col="filenames",
                                                                  y_col="log_B19013001",
                                                                  target_size=(224, 224),
                                                                  color_mode="rgb",
                                                                  class_mode="raw",
                                                                  batch_size=batch_size,
                                                                  shuffle=True,
                                                                  seed=123)
    test_generator = test_datagen.flow_from_dataframe(dataframe=test_label_df,
                                                      directory="data/street_view/Los Angeles/images",
                                                      x_col="filenames",
                                                      y_col="log_B19013001",
                                                      target_size=(224, 224),
                                                      color_mode="rgb",
                                                      class_mode="raw",
                                                      batch_size=batch_size,
                                                      shuffle=False)

    # Train model
    model = build_model()
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_label_df) // batch_size,
                        epochs=7,
                        verbose=1,
                        validation_data=validation_generator,
                        validation_steps=len(validation_label_df) // batch_size)
    train_generator.reset()
    validation_generator.reset()

    # Get training results
    history_dict = history.history
    history_dict.keys()
    epochs = range(1, len(history_dict["loss"]) + 1)

    # Plot training/validation loss
    plt.plot(epochs, history_dict["loss"], label="training loss")
    plt.plot(epochs, history_dict["val_loss"],
             color="tab:orange",
             label="validation loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    # # Retrain model
    # model = build_model()
    # model = build_model()
    # model.fit(train_generator,
    #           steps_per_epoch=len(train_label_df) // batch_size,
    #           epochs=10,
    #           verbose=1,
    #           validation_data=validation_generator,
    #           validation_steps=len(validation_label_df) // batch_size)

    # Make predictions
    y_test_hat = model.predict(test_generator,
                               steps=len(test_label_df) // batch_size,
                               verbose=1)
    test_generator.reset()
    print(calc_R_2(y_test_hat, test_label_df["log_B19013001"].values))

    # Plot actual vs predicted
    plt.figure()
    plt.scatter(scaler.inverse_transform(y_test_hat),
                scaler.inverse_transform(test_label_df["log_B19013001"][:len(y_test_hat)]),
                color="#4A89F3",
                s=1)  # marker size
    plt.xlabel("Predicted Log Income")
    plt.ylabel("Actual Log Income")
    # plt.xlim(8, 12)
    # plt.ylim(8, 12)
    # plt.savefig("model visualizations/models_deep.png", dpi=200,
    #             bbox_inches="tight")
    plt.show()

    # Save weights
    model.save("models_deep.h5")
