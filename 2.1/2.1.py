import numpy
import pandas


# Hipoteza: funkcja liniowa jednej zmiennej
def h(theta, x):
    return theta[0] + theta[1] * x


# Funkcja kosztu
def J(h, theta, x, y):
    m = len(y)
    return 1.0 / (2 * m) * sum((h(theta, x[i]) - y[i])**2 for i in range(m))


# Wykres funkcji kosztu dla ustalonego theta_1=1.0
def cost_fun(fun, x, y):
    return lambda theta: J(fun, theta, x, y)


# Oblicz parametry θ krzywej regresyjnej za pomocą metody gradientu prostego
# (gradient descent). Możesz wybrać wersję iteracyjną lub macierzową algorytmu.
def gradient_descent(h, cost_fun, theta, x, y, alpha, eps):
    current_cost = cost_fun(h, theta, x, y)
    log = [[current_cost, theta]]  # log przechowuje wartości kosztu i parametrów
    m = len(y)
    while True:
        new_theta = [
            theta[0] - alpha/float(m) * sum(h(theta, x[i]) - y[i]
                                            for i in range(m)),
            theta[1] - alpha/float(m) * sum((h(theta, x[i]) - y[i]) * x[i]
                                            for i in range(m))]
        theta = new_theta  # jednoczesna aktualizacja - używamy zmiennej tymaczasowej
        try:
            current_cost, prev_cost = cost_fun(h, theta, x, y), current_cost
        except OverflowError:
            break
        if abs(prev_cost - current_cost) <= eps:
            break
        log.append([current_cost, theta])
    return theta, log


def mse(expected, predicted):
    """Błąd średniokwadratowy"""
    m = len(expected)
    if len(predicted) != m:
        raise Exception('Wektory mają różne długości!')
    return 1.0 / (2 * m) * sum((expected[i] - predicted[i]) ** 2 for i in range(m))


# https://lms.amu.edu.pl/sci/mod/assign/view.php?id=6333
# https://git.wmi.amu.edu.pl/pms/zuma/src/branch/master/wyk/2_Regresja_liniowa.ipynb
def zadanie_2_1():
    # liczba pożarów w danej dzielnicy na tysiąc gospodarstw domowych (pierwsza kolumna)
    # oraz liczba włamań w tej samej dzielnicy na tysiąc mieszkańców (druga kolumna)

    data = pandas.read_csv('fires_thefts.csv', header=None)
    learn_set = 0.8
    test_set = 1 - learn_set

    x_learn = data[0][:int(len(data[0]) * learn_set)]
    y_learn = data[1][:int(len(data[1]) * learn_set)]

    # Stwórz model (regresja liniowa) przewidujący
    # liczbę włamań na podstawie liczby pożarów:

    # Poeksperymentuj z różnymi wartościami współczynnika szybkości uczenia α:
    alpha = 0.0001

    eps = 0.00001
    theta = [0.0, 0.0]
    best_theta, log = gradient_descent(h, J, theta, x_learn, y_learn, alpha, eps)

    # # Obliczenie MSE na zbiorze testowym (im mniejszy MSE, tym lepiej!)
    x_test = data[0][-int(len(data[0]) * test_set):]
    y_test = h(best_theta, x_test)
    y_learn = data[1][-len(y_test):]

    # metoda gradientu nie przyjmuje listy numpy tylko listę python
    y_test = list(y_test)
    y_learn = list(y_learn)

    evaluation_result = mse(y_learn, y_test)
    print("evaluation: %s" % evaluation_result)
    print("best_theta: %s" % best_theta)

    # Wykorzystując uzyskaną krzywą regresyjną przepowiedz liczbę włamań na tysiąc
    # mieszkańców dla dzielnicy, w której występuje średnio 50, 100, 200 pożarów
    # na tysiąc gospodarstw domowych.
    x = numpy.array([50, 100, 200])
    print("prediction for %s: %s" % (x, h(best_theta, x)))


if __name__ == '__main__':
    zadanie_2_1()
