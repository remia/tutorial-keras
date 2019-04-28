import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers


np.set_printoptions(suppress=True)


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5, batch_size=128)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc : ', test_acc)

    for filename in os.listdir('data'):
        if not filename.endswith('png'):
            continue

        img = imread(os.path.join('data', filename))
        img = img[:, :, 0]
        img = 255 - img

        pred_data = np.zeros((1, 28 * 28))
        pred_data[0, :] = img.reshape(28 * 28).astype('float32') / 255
        prediction = model.predict(pred_data)
        print(filename, '===>', prediction.argmax())
