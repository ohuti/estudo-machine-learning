from numpy import arange
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from matplotlib.pyplot import scatter, show, plot


def main():
    x, y = make_regression(n_samples=200, n_features=1, noise=30)

    scatter(x, y)
    show()

    model = LinearRegression()
    model.fit(x, y)

    print(model.intercept_)
    print(model.coef_)

    m = model.coef_[0]
    b = model.intercept_

    scatter(x, y)
    xreg = arange(-3, 4, 1)
    plot(xreg, m*xreg+b, color='red')
    show()


if __name__ == '__main__':
    main()
