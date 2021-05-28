import os
import numpy as np
import pandas as pd
from keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import vgg16
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# Instantiate the VGG16 model
model = VGG16(include_top=True, weights="imagenet")
extractor = Model(inputs=model.inputs, outputs=model.get_layer("fc2").output)

# Plot model architecture
plot_model(model, to_file="model visualizations/vgg16_architecture.png",
           show_layer_names=True,
           rankdir="TB",  # "TB" creates a vertical plot, "LR" is horizontal
           dpi=240)

# Dataframe to store itermediate outputs
filenames = os.listdir("./data/street_view/New York/images")
n = len(filenames)
X = np.zeros(shape=(n, 4096), dtype="float32")
X = pd.DataFrame(data=X, index=filenames)
X["filenames"] = X.index

batch_size = 25
steps = len(X) // 40

datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
image_generator = datagen.flow_from_dataframe(dataframe=X,
                                              directory="./data/street_view/New York/images",
                                              x_col="filenames",
                                              target_size=(224, 224),
                                              color_mode="rgb",
                                              class_mode=None,
                                              batch_size=batch_size,
                                              shuffle=False)

for step, batch in zip(range(steps), image_generator):
    start = time.time()
    test = batch
    X.iloc[step*batch_size:(step+1)*batch_size, :4096] = extractor(batch).numpy()
    print("#" + str(step + 1), time.time() - start)
image_generator.reset()

# Write outputs to .csv files
X.drop(columns="filenames", inplace=True)
X.to_csv("X_ny_vgg16.csv")
