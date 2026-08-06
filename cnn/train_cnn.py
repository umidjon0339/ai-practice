#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================
# Train a simple Convolutional Neural Network (CNN) on the CIFAR-10 dataset.
#
# CIFAR-10 = 60,000 tiny color images (32x32 pixels), 10 classes,
# 6,000 images per class. 50,000 are for training, 10,000 for testing.
#
# Run from the project root:  python cnn/train_cnn.py
# ============================================================================

import tensorflow as tf                                  # the ML framework itself
from tensorflow.keras import datasets, layers, models    # dataset loader + building blocks
import matplotlib.pyplot as plt                          # for drawing plots

# The dataset stores labels as numbers 0..9. This list translates a number
# into a human-readable name, e.g. label 3 -> class_names[3] -> 'cat'.
# The ORDER matters and must never change - the model only knows the numbers.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def load_data():
    """Download CIFAR-10 and normalize pixel values to [0, 1]."""
    # First call downloads ~170 MB; later calls load from the
    # ~/.keras/datasets cache instantly.
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # NORMALIZATION: pixels come as integers 0..255. Dividing by 255 rescales
    # them to 0..1. Neural networks train much better on small consistent
    # ranges. IMPORTANT: any image we predict on later (see app.py) must be
    # preprocessed the exact same way, or predictions become garbage.
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def show_samples(train_images, train_labels):
    """Plot the first 25 training images with their class names.

    This is only a sanity check that the data looks right - it does not
    affect training in any way.
    """
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)   # position i+1 in a 5x5 grid
        plt.xticks([])             # hide axis ticks - they mean nothing here
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # Each label is stored as a 1-element array like [6], not a plain
        # number - that is why the extra [0] index is needed.
        plt.xlabel(class_names[train_labels[i][0]])
    plt.savefig('cnn/samples.png')
    plt.show()


def build_model():
    """Build the convolutional base plus dense classification head."""
    model = models.Sequential([
        # Input: 32x32 pixels, 3 color channels (RGB). Batch size is not
        # part of the shape - the model accepts any number of images at once.
        layers.Input(shape=(32, 32, 3)),

        # --- Convolutional base: learns WHAT is in the image ---
        # Conv2D slides 32 different 3x3 filters across the image. Each
        # filter learns to detect one simple pattern (an edge, a color blob).
        # 'relu' = keep positive values, zero out negatives. Without this
        # non-linearity, stacking layers would be mathematically pointless.
        layers.Conv2D(32, (3, 3), activation='relu'),

        # MaxPooling halves the width/height by keeping only the strongest
        # value in each 2x2 square. Makes the net faster and more tolerant
        # to small shifts of the object in the image.
        layers.MaxPooling2D((2, 2)),

        # Deeper conv layers combine simple patterns into complex ones
        # (edges -> textures -> object parts). As the image shrinks we can
        # afford more filters (32 -> 64).
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),

        # --- Classification head: decides WHICH class it is ---
        # The conv base ends with a 4x4x64 grid. Dense layers only accept
        # flat vectors, so Flatten unrolls it into 1024 numbers.
        layers.Flatten(),
        layers.Dense(64, activation='relu'),   # mixes all features together

        # Output: one raw score per class. These are called LOGITS -
        # not probabilities yet (softmax is applied later, see the loss).
        layers.Dense(10),
    ])

    # compile() configures HOW to train:
    model.compile(
        # adam = the algorithm that nudges the weights after each batch.
        optimizer='adam',
        # The loss = the "wrongness score" that training minimizes.
        # 'Sparse'      -> labels are plain integers (3), not one-hot vectors.
        # from_logits   -> model outputs raw scores, so softmax is applied
        #                  inside the loss function automatically.
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # Also report accuracy (% correct) - easier for humans than loss.
        metrics=['accuracy'])
    return model


def plot_history(history):
    """Plot training and validation accuracy per epoch.

    How to read it: if the training curve keeps rising while the validation
    curve flattens or drops, the model is memorizing the training set
    instead of learning general patterns (overfitting).
    """
    plt.figure()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig('cnn/training_history.png')
    plt.show()


def main():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    show_samples(train_images, train_labels)

    model = build_model()
    model.summary()   # print the layer-by-layer architecture table

    # fit() is the actual training loop: show the model all 50,000 images
    # 10 times (epochs), in batches of 32, adjusting weights after every
    # batch. validation_data is evaluated after each epoch so we can watch
    # val_accuracy - the model never trains on it.
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    plot_history(history)

    # Final, definitive score on images the model has never trained on.
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Test accuracy: {test_acc}')

    # Save the learned weights + architecture to one file. This is exactly
    # the file that app.py loads - the app never trains anything, it just
    # replays these frozen weights on new images.
    model.save('cnn/cifar10_cnn.keras')


if __name__ == '__main__':
    main()
