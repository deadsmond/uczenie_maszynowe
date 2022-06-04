import pandas as pd
import matplotlib.pyplot as plt


def zadanie_1_6():
    data = pd.read_csv('data2.csv')
    print(data)

    # Wczytanie danych
    data_array = data.to_numpy()

    # Wybór kolumn do przedstawienia na wykresie
    x = data_array[:, 2]
    y = data_array[:, 3]

    # pokaż wykres
    plt.plot(x, y, 'ro')
    plt.show()


if __name__ == '__main__':
    zadanie_1_6()
