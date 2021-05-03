from tensorflow.keras import models
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def cnn_visualize(model, layers=8):
    # Visualize intermmediate activations
    layer_outputs = [layer.output for layer in model.layers[:layers]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    for i in range(len(layer_outputs)):
        activation = activations[i]
        for j in range(activation.shape[3]):
            plt.imshow(activation[0, :, :, j], cmap="viridis")
            plt.axis("off")
            plt.title(f"Layer: {i}, Channel: {j}")
            plt.show()


if __name__ == "__main__":
    # Load saved model
    model = models.load_model("cnn_3epoch.h5")

    # Plot all activations
    # cnn_visualize(model, layers=8)

    # Define intermediate network
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    # Load sample image
    img = np.flip(cv.imread("data/street_view/Los Angeles/images/15000US060371011101_1.jpeg"),
                  axis=2)
    img_tensor = np.expand_dims(img, axis=0)
    activations = activation_model.predict(img_tensor)
    # Plot original image and two activations
    fig, axes = plt.subplots(1, 3, gridspec_kw={"wspace": 0, "hspace": 0})
    # Original image
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Input Image")
    # Layer 2 Channel 9
    activation = activations[2]
    axes[1].imshow(activation[0, :, :, 9], cmap="viridis")
    axes[1].axis("off")
    axes[1].set_title("Layer 2")
    # Layer 5 Channel 56 activation
    activation = activations[5]
    axes[2].imshow(activation[0, :, :, 56], cmap="viridis")
    axes[2].axis("off")
    axes[2].set_title("Layer 5")
    # plt.savefig("model visualizations/cnn1_intermediate.png", dpi=480, bbox_inches="tight")
    plt.show()
