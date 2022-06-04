# https://github.com/pellway/keras_mushroom_model
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


def zadanie_4():
    # Load mushroom dataset
    column_names = ['class',
                    'cap-shape',
                    'cap-surface',
                    'cap-color',
                    'bruises?',
                    'odor',
                    'gill-attachment',
                    'gill-spacing',
                    'gill-size',
                    'gill-color',
                    'stalk-shape',
                    'stalk-root',
                    'stalk-surface-above-ring',
                    'stalk-surface-below-ring',
                    'stalk-color-above-ring',
                    'stalk-color-below-ring',
                    'veil-type',
                    'veil-color',
                    'ring-number',
                    'ring-type',
                    'spore-print-color',
                    'population',
                    'habitat']
    data = pd.DataFrame(pd.read_csv('mushrooms.tsv', sep="\t", header=None, names=column_names))

    learn_set = 0.9
    test_set = 1 - learn_set

    # Convert string data into dummy integer
    data = data.dropna()
    data = pd.get_dummies(data)

    # split into input (X) and output (y) variables
    x = data.drop(columns=[0])
    y = data[[0]].copy()

    # Create Keras model using layers
    model = Sequential()
    model.add(Dense(18, input_dim=22, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit Keras model on dataset
    model.fit(x, y, validation_split=test_set, epochs=30, batch_size=10, verbose=0)

    # Evaluate Keras model
    # accuracy:
    accuracy = model.evaluate(x, y)
    # score
    score = accuracy[1]
    print('Program completed with Accuracy: %.2f' % (score*100) + '%')


if __name__ == '__main__':
    zadanie_4()
