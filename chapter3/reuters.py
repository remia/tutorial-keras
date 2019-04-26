import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras import models
from keras import layers


def decode_review(item):
    word_index = reuters.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in item])


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def to_one_hot(labels, dimensions=46):
    results = np.zeros((len(labels), dimensions))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


def plot_loss(loss_values, val_loss_values):
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_accuracy(acc, val_acc):
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def show_random_results(count, test_data, x_test, predictions):
    indices = np.random.randint(0, test_data.shape[0], 5)
    for i in indices:
        print("Review : ", decode_review(test_data[i]))
        print("Review vector : ", x_test[i])
        print("Model prediction : ", predictions[i])


if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    # Reset numpy seed state because reuters.load_data used fixed seed by default
    np.random.seed()

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    # Could also have used :
    # from keras.utils.np_utils import to_categorical
    one_hot_train_labels = to_one_hot(train_labels)
    one_hot_test_labels = to_one_hot(test_labels)

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=9,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    predictions = model.predict(x_test)
    evaluations = model.evaluate(x_test, one_hot_test_labels)
    print("Test dataset", evaluations)

    history_dict = history.history
    plot_loss(history_dict['loss'], history_dict['val_loss'])
    plt.clf()
    plot_accuracy(history_dict['acc'], history_dict['val_acc'])

    # show_random_results(5, test_data, x_test, predictions)
