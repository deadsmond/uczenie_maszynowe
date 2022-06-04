import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge


def solve(degree: int, x, y, color, label, model):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(x.reshape(-1, 1))
    poly_reg_model = model
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)
    plt.plot(x, y_predicted, c=color, label=label)


def zadanie_3_2():

    # Plik data6.tsv zawiera pewne dane. Zastosuj do nich regresję wielomianową:
    data = pandas.read_csv('data6.tsv', sep='\t', header=None)
    data.sort_values(data.columns[0], inplace=True)
    x = data.iloc[:, 0].to_numpy()
    y = data.iloc[:, 1].tolist()
    
    plt.scatter(x, y)

    # pierwszego stopnia (funkcja liniowa)
    solve(1, x, y, "red", "1 dg", LinearRegression())

    # drugiego stopnia (funkcja kwadratowa)
    solve(2, x, y, "yellow", "2 dg", LinearRegression())

    # piątego stopnia (wielomian 5. stopnia)
    solve(5, x, y, "green", "5 dg", LinearRegression())

    # piątego stopnia z regularyzacją
    solve(5, x, y, "black", "5 dg w/ reg", Ridge(alpha = 0.5))

    # Otrzymane krzywe regresji przedstaw na wykresie.
    plt.legend()
    plt.show()


if __name__ == '__main__':
    zadanie_3_2()
