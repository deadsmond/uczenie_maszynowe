import pandas
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# https://lms.amu.edu.pl/sci/mod/assign/view.php?id=6333
# https://git.wmi.amu.edu.pl/pms/zuma/src/branch/master/wyk/2_Regresja_liniowa.ipynb
def zadanie_2_2():
    # liczba pożarów w danej dzielnicy na tysiąc gospodarstw domowych (pierwsza kolumna)
    # oraz liczba włamań w tej samej dzielnicy na tysiąc mieszkańców (druga kolumna)

    data = pandas.read_csv('fires_thefts.csv', header=None)
    learn_set = 0.8
    test_set = 1 - learn_set

    x_learn = data[0][:int(len(data[0]) * learn_set)].values.reshape(-1, 1)
    y_learn = data[1][:int(len(data[1]) * learn_set)].values.reshape(-1, 1)

    # Stwórz model (regresja liniowa) przewidujący
    # liczbę włamań na podstawie liczby pożarów:

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_learn, y_learn)

    # Make predictions using the testing set
    x_test = data[0][-int(len(data[0]) * test_set):].values.reshape(-1, 1)
    y_test = data[1][-len(x_test):].values.reshape(-1, 1)
    y_pred = regr.predict(x_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Wykorzystując uzyskaną krzywą regresyjną przepowiedz liczbę włamań na tysiąc
    # mieszkańców dla dzielnicy, w której występuje średnio 50, 100, 200 pożarów
    # na tysiąc gospodarstw domowych.
    x = np.array([50, 100, 200]).reshape(-1, 1)
    print("prediction for %s: %s" % (x, regr.predict(x)))


if __name__ == '__main__':
    zadanie_2_2()
