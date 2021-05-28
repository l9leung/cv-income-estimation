import os
# from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import vgg16
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications import resnet
from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Average
# from tensorflow.keras.layers import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from data.acs_load import acs_load
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# def build_model():
#     model = Sequential()
#     vgg16_model = VGG16(include_top=True, weights="imagenet")
#     for layer in vgg16_model.layers[:-1]:
#         model.add(layer)
#     for layer in model.layers:
#         layer.trainable = False
#     model.add(Dense(512, activation="relu", input_shape=(4096,)))
#     model.add(Dense(256, activation="relu"))
#     model.add(Dense(128, activation="relu"))
#     model.add(Dense(64, activation="relu"))
#     model.add(Dense(1))
#     model.compile(optimizer=Adam(learning_rate=0.001), loss="mse",
#                   metrics="mae")
#     return model


def build_model(base="VGG16", trainable=False):
    inputs = {}
    layers = {}
    denses = {}
    for input_number in range(1, 6):
        inputs[input_number] = Input(shape=(224, 224, 3))
        last_layer = inputs[input_number]
        if base == "VGG16":
            vgg16_model = VGG16(include_top=True, weights="imagenet")
            for layer in vgg16_model.layers[:-1]:
                if trainable is False:
                    layer.trainable = False
                layer._name = f"{layer._name}_{input_number}"
                layers[f"{layer._name}_{input_number}"] = layer(last_layer)
                last_layer = layers[f"{layer._name}_{input_number}"]
            denses[f"dense_{input_number}_1"] = Dense(512, activation="relu", input_shape=(4096,))(last_layer)
        # elif base == "ResNet50":
        #     resnet50_model = ResNet50(include_top=True, weights="imagenet")
        #     resnet50_model._name = f"resnet50_{input_number}"
        #     for layer in resnet50_model.layers[:-1]:
        #         if trainable is False:
        #             layer.trainable = False
        #         layer._name = f"{layer._name}_{input_number}"
        #     layers[f"resnet50_{input_number}"] = resnet50_model(last_layer)
        #     denses[f"dense_{input_number}_1"] = Dense(512, activation="relu", input_shape=(2048,))(resnet50_model.layers[-2].output)
        denses[f"dense_{input_number}_2"] = Dense(256, activation="relu")(denses[f"dense_{input_number}_1"])
        denses[f"dense_{input_number}_3"] = Dense(128, activation="relu")(denses[f"dense_{input_number}_2"])
    # merged = Add()([denses["dense_1_3"], denses["dense_2_3"], denses["dense_3_3"], denses["dense_4_3"], denses["dense_5_3"]])
    merged = Average()([denses["dense_1_3"],
                        denses["dense_2_3"],
                        denses["dense_3_3"],
                        denses["dense_4_3"],
                        denses["dense_5_3"]])
    dense1 = Dense(64, activation="relu")(merged)
    output = Dense(1)(dense1)

    model = Model(inputs=[inputs[input_number] for input_number in range(1, 6)],
                  outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse",
                  metrics="mae")

    return model


def get_flow_from_dataframe(datagen,
                            city,
                            dataframe,
                            batch_size=50):
    generator_1 = datagen.flow_from_dataframe(dataframe=dataframe,
                                              directory=f"data/street_view/{city}/images",
                                              x_col="0",
                                              y_col="log_B19013001",
                                              target_size=(224, 224),
                                              color_mode="rgb",
                                              class_mode="raw",
                                              batch_size=batch_size,
                                              shuffle=True,
                                              seed=123)
    generator_2 = datagen.flow_from_dataframe(dataframe=dataframe,
                                              directory=f"data/street_view/{city}/images",
                                              x_col="1",
                                              y_col="log_B19013001",
                                              target_size=(224, 224),
                                              color_mode="rgb",
                                              class_mode="raw",
                                              batch_size=batch_size,
                                              shuffle=True,
                                              seed=123)
    generator_3 = datagen.flow_from_dataframe(dataframe=dataframe,
                                              directory=f"data/street_view/{city}/images",
                                              x_col="2",
                                              y_col="log_B19013001",
                                              target_size=(224, 224),
                                              color_mode="rgb",
                                              class_mode="raw",
                                              batch_size=batch_size,
                                              shuffle=True,
                                              seed=123)
    generator_4 = datagen.flow_from_dataframe(dataframe=dataframe,
                                              directory=f"data/street_view/{city}/images",
                                              x_col="3",
                                              y_col="log_B19013001",
                                              target_size=(224, 224),
                                              color_mode="rgb",
                                              class_mode="raw",
                                              batch_size=batch_size,
                                              shuffle=True,
                                              seed=123)
    generator_5 = datagen.flow_from_dataframe(dataframe=dataframe,
                                              directory=f"data/street_view/{city}/images",
                                              x_col="4",
                                              y_col="log_B19013001",
                                              target_size=(224, 224),
                                              color_mode="rgb",
                                              class_mode="raw",
                                              batch_size=batch_size,
                                              shuffle=True,
                                              seed=123)
    while True:
        X_1 = generator_1.next()
        X_2 = generator_2.next()
        X_3 = generator_3.next()
        X_4 = generator_4.next()
        X_5 = generator_5.next()

        yield [X_1[0], X_2[0], X_3[0], X_4[0], X_5[0]], X_1[1]


def calc_R_2(y_hat, y_test):
    """Returns the coefficient of determination $R^2$ given predicted values
    y_hat and true values y_test."""
    SSR = ((y_test - y_hat)**2).sum()
    SST = ((y_test - y_test.mean())**2).sum()

    return 1 - (SSR/SST)


if __name__ == "__main__":
    city = "New York"
    base = "VGG16"

    # Image file names
    label_df = pd.DataFrame({"filenames": os.listdir(f"./data/street_view/{city}/images")})
    label_df["geoid"] = label_df["filenames"].str.split("_", expand=True)[0]
    # Blocks with 5 images
    full = label_df.groupby("geoid").count()[label_df.groupby("geoid").count()["filenames"] == 5].index.to_list()
    label_df = label_df[label_df["geoid"].isin(full)]
    del full
    label_df["i"] = label_df["filenames"].str.split("_", expand=True)[1].str.split(".", expand=True)[0]
    label_df = label_df.pivot(index="geoid", columns="i", values="filenames")
    # Block group median income
    acs = acs_load(cities=[city])[city]

    # Drop outliers
    # acs["B19013001_thousand"].median() - 2*acs["B19013001_thousand"].std()
    # acs["B19013001_thousand"].median() + 2*acs["B19013001_thousand"].std()
    # acs = acs[(acs["B19013001_thousand"] > 30) & (acs["B19013001_thousand"] < 150)]
    acs = acs[acs["B19013001_thousand"] < 150]
    acs = acs[["geoid", "log_B19013001"]]

    # Join file names with median incomes
    label_df = label_df.merge(acs, how="inner", left_index=True, right_on="geoid")
    del acs

    # Drop images with missing median incomes
    label_df = label_df.dropna()

    # Shuffle dataframe
    label_df = label_df.sample(frac=0.5, random_state=123).reset_index(drop=False)

    # Sample splitting
    train_label_df = label_df[:2000]
    test_label_df = label_df[2000:].copy()
    validation_label_df = train_label_df[1500:].copy()
    train_label_df = train_label_df[:1500]
    del label_df

    # Standardize log income using training set range
    scaler = StandardScaler().fit(train_label_df[["log_B19013001"]])
    train_label_df["log_B19013001"] = scaler.transform(train_label_df[["log_B19013001"]].astype("float32"))
    validation_label_df["log_B19013001"] = scaler.transform(validation_label_df[["log_B19013001"]].astype("float32"))
    test_label_df["log_B19013001"] = scaler.transform(test_label_df[["log_B19013001"]].astype("float32"))

    # Image data generators
    batch_size = 25
    if base == "VGG16":
        datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
    # elif base == "ResNet50":
    #     datagen = ImageDataGenerator(preprocessing_function=resnet.preprocess_input)
    train_generator = get_flow_from_dataframe(datagen,
                                              city=city,
                                              dataframe=train_label_df,
                                              batch_size=batch_size)
    validation_generator = get_flow_from_dataframe(datagen,
                                                   city=city,
                                                   dataframe=validation_label_df,
                                                   batch_size=batch_size)
    test_generator = get_flow_from_dataframe(datagen,
                                             city=city,
                                             dataframe=test_label_df,
                                             batch_size=batch_size)

    # Get model
    model = build_model(base, trainable=False)

    # Plot model
    plot_model(model, to_file=f"model visualizations/{base}_architecture.png",
               show_layer_names=True,
               rankdir="TB"  # "TB" creates a vertical plot, "LR" is horizontal
               )

    # Train model
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_label_df) // batch_size,
                        epochs=5,
                        verbose=1,
                        validation_data=validation_generator,
                        validation_steps=len(validation_label_df) // batch_size)
    # train_generator.reset()
    # validation_generator.reset()

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
    # test_generator.reset()
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
