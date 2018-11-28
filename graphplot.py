import keras
import imageio
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def plot_labels_(subplot, xlabel='', ylabel=''):
    if xlabel != '':
        subplot.set_xlabel(xlabel)
    if ylabel != '':
        subplot.set_ylabel(ylabel)

def plot_signal_(subplot, data, xlabel='', ylabel=''):
    assert len(data)
    plot_labels_(subplot, xlabel, ylabel)
    subplot.plot(data)

def plot_specgram_(subplot, data, freq, xlabel='', ylabel=''):
    assert len(data)
    assert freq
    plot_labels_(subplot, xlabel, ylabel)
    plt.specgram(data,Fs=freq)

def plot_data(freq, data):
    fig = plt.figure(1)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    plot_signal_(
            subplot=ax1,
            data=data,
            xlabel='Sample',
            ylabel='Amplitude')
    plot_specgram_(
            subplot=ax2,
            data=data,
            freq=freq,
            xlabel='Time',
            ylabel='Frequency')
    plt.show()

def save_specgram(freq, data, filename):
    fig = plt.figure(1)
    plt.axis('off')
    plot_specgram_(
            subplot=fig,
            data=data,
            freq=freq)
    plt.savefig(filename,
            bbox_inches='tight',
            pad_inches=0.0,
            transparent=True)

def make_model():
    # Inspired from VGG16 but with fewer layers and
    # pool layers with larger pool_size
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64,
                            input_shape=(224,224,3),
                            kernel_size=(3,3),
                            padding='same',
                            activation='relu'))
    model.add(keras.layers.Conv2D(filters=64,
                            kernel_size=(3,3),
                            padding='same',
                            activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(4,4),
                            strides=(2,2)))
    model.add(keras.layers.Conv2D(filters=128,
                            kernel_size=(3,3),
                            padding='same',
                            activation='relu'))
    model.add(keras.layers.Conv2D(filters=128,
                            kernel_size=(3,3),
                            padding='same',
                            activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(4,4),
                            strides=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    return model

def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.SGD(
                        lr=0.01,
                        momentum=0.8,
                        nesterov=True
                    ))
    return model

def load_data(files):
    x_train = []
    y_train = []
    for filename in files:
        image = imageio.imread(filename)
        x_train.append(image)
        # resize image to 224x224x3
        # y_labels are 1 or 0 based on if it is
        # the sound of horn or not
        y_train.append(1)
    return (np.array(x_train), np.array(y_train))

def main():
    # sample_rate, samples = wavfile.read('cars014.wav')
    # plt.set_cmap('hot')
    # # plot_data(sample_rate, samples)
    # save_specgram(sample_rate, samples, 'cars014.png')
    model = compile_model(make_model())
    # pass list of images as an iterable. can use glob
    x_train, y_train = load_data([
            'cars014.png', 'cars014.png', 'cars014.png',
            'cars014.png', 'cars014.png', 'cars014.png',
            'cars014.png', 'cars014.png', 'cars014.png',
            'cars014.png', 'cars014.png', 'cars014.png',
            'cars014.png', 'cars014.png', 'cars014.png',
            'cars014.png', 'cars014.png', 'cars014.png',
            'cars014.png', 'cars014.png', 'cars014.png',
            'cars014.png', 'cars014.png', 'cars014.png',
            'cars014.png', 'cars014.png', 'cars014.png',
            'cars014.png', 'cars014.png', 'cars014.png',
            'cars014.png', 'cars014.png'])
    model.fit(x_train, y_train, epochs=5, batch_size=32)

if __name__ == '__main__':
    main()
