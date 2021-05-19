from time import time
from numpy import arange
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import scatter, show, plot, title


def main():
    start_time = time()

    x, y = make_regression(n_samples=1200, n_features=1, noise=25)

    x_practice, x_test, y_practice, y_test = train_test_split(x, y, test_size=0.2)

    model = LinearRegression()
    model.fit(x_practice, y_practice)

    m = model.coef_[0]
    b = model.intercept_

    x_min = round(min(row[0] for row in x_test), 0)
    x_max = round(max(row[0] for row in x_test), 0)

    finish_time = time()

    elapsed_time = finish_time - start_time
    print(f'Score da regressão: {model.score(x_test, y_test)}.')
    print(f'Tempo de execução: {elapsed_time}s.')

    scatter(x_test, y_test)
    xreg = arange(x_min, x_max, 1)
    plot(xreg, m*xreg+b, color='red')
    title(f'Qtd. dados de treino: {len(x_practice)}\nQtd. dados de teste: {len(x_test)}\nScore da regressão: {model.score(x_test, y_test)}.')
    show()


if __name__ == '__main__':
    main()
