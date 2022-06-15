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

    learn_set = 0.75

    # get dummies numbers
    data = data.dropna()
    data = pd.get_dummies(data)

    # split into input (x) and expected output (y)
    x = data.drop(columns=["class_e", "class_p"])
    y = data[["class_e"]].copy()

    # Create model
    model = Sequential()
    model.add(Dense(18, input_dim=117, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model using the training sets
    model.fit(x, y, validation_split=(1 - learn_set), epochs=30, batch_size=10, verbose=0)

    # evaluate model:
    model.summary()
    evaluate = model.evaluate(x, y)
    print('loss value: %.10f' % evaluate[0])
    print('score: %.2f' % (evaluate[1]*100) + '%')


if __name__ == '__main__':
    zadanie_4()
