from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt


def calc_R_2(y_hat, y_test):
    """Returns the coefficient of determination R^2 given predicted values
    y_hat and ground truth labels y_test."""
    SSR = ((y_test - y_hat)**2).sum()
    SST = ((y_test - y_test.mean())**2).sum()

    return 1 - (SSR/SST)


# Model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu",
                        input_shape=(400, 600, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1))
model.compile(loss="mse", optimizer=optimizers.Adam(), metrics=["mae"])

# Plot model
plot_model(model, to_file="model visualizations/cnn1_architecture.png", show_layer_names=True,
           rankdir="TB",  # "TB" creates a vertical plot, "LR" is horizontal
           dpi=240)

# Image file names
label_df = pd.DataFrame({"image_name": os.listdir("./data/street_view/Los Angeles/images")})
label_df["geoid"] = label_df["image_name"].str.split("_", expand=True)[0]
# Block group median income
incomes = pd.read_csv("./data/acs/Los Angeles/16000US0644000_block_household_income.csv",
                      index_col=0, usecols=[0, 3])
# Join file names with median incomes
label_df = label_df.merge(incomes, how="inner",
                          left_on="geoid", right_index=True)
# Drop images with missing median incomes
label_df = label_df.dropna()
# Log transform income
label_df["log_estimate"] = np.log(label_df["estimate"])
# Shuffle dataframe
label_df = label_df.sample(frac=1, random_state=123).reset_index(drop=True)
# Use 19000 of samples for training and validation, 5290 for testing
train_label_df = label_df[:19000]
test_label_df  = label_df[19000:]
# Minmax transform income using training set range
scaler = MinMaxScaler().fit(train_label_df[["log_estimate"]])
train_label_df["estimate_scaled"] = scaler.transform(train_label_df[["log_estimate"]])
test_label_df["estimate_scaled"] = scaler.transform(test_label_df[["log_estimate"]])
# Use 5000 of training samples for validation
validation_label_df = train_label_df[14000:19000]
train_label_df = train_label_df[:14000]

# Image data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
batch_size = 50
train_generator = train_datagen.flow_from_dataframe(dataframe=train_label_df,
                                                    directory="data/street_view/Los Angeles/images",
                                                    x_col="image_name",
                                                    y_col="estimate_scaled",
                                                    target_size=(400, 600),
                                                    color_mode="rgb",
                                                    class_mode="raw",
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    seed=123)
validation_generator = validation_datagen.flow_from_dataframe(dataframe=validation_label_df,
                                                              directory="data/street_view/Los Angeles/images",
                                                              x_col="image_name",
                                                              y_col="estimate_scaled",
                                                              target_size=(400, 600),
                                                              color_mode="rgb",
                                                              class_mode="raw",
                                                              batch_size=batch_size,
                                                              shuffle=True,
                                                              seed=123)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_label_df,
                                                  directory="data/street_view/Los Angeles/images",
                                                  x_col="image_name",
                                                  y_col="estimate_scaled",
                                                  target_size=(400, 600),
                                                  color_mode="rgb",
                                                  class_mode="raw",
                                                  batch_size=batch_size,
                                                  shuffle=False)

# Train model
history = model.fit(train_generator,
                    steps_per_epoch=len(train_label_df) // batch_size,
                    epochs=5,
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps=len(validation_label_df) // batch_size)
history_dict = history.history
epochs = range(1, len(history_dict["loss"]) + 1)
# Plot model loss
plt.plot(epochs, history_dict["loss"], label="Training Loss")
plt.plot(epochs, history_dict["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xticks(epochs)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Save weights
model.save("cnn_5epoch.h5")

# Load model
# model = models.load_model("cnn_5epoch.h5")

# Make predictions
predictions = model.predict(test_generator,
                            steps=len(test_label_df) // batch_size,
                            verbose=1)
test_generator.reset()

# Evaluate model on test set
results = pd.DataFrame({"image_name": test_label_df[:len(test_label_df) - len(test_label_df)%batch_size]["image_name"],
                        "predicted": predictions.ravel(),
                        "actual": test_label_df[:len(test_label_df) - len(test_label_df)%batch_size]["estimate_scaled"]})
# Calculate R^2
calc_R_2(results["predicted"].ravel(),
         results["actual"].ravel())

# Plot actual vs predicted
plt.figure()
plt.scatter(scaler.inverse_transform(results[["predicted"]]),
            scaler.inverse_transform(results[["actual"]]),
            s=1)  # marker size
plt.xlabel("Predicted Log Income")
plt.ylabel("Actual Log Income")
plt.savefig("predictedvsactual_5epochs.png", dpi=360, bbox_inches="tight")
plt.show()
