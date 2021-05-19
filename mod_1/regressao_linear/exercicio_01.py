import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    file = pd.read_csv('source_files/kc_house_data.csv')

    pd.set_option('display.max_columns', len(file.columns))

    file.drop('id', axis=1, inplace=True)
    file.drop('date', axis=1, inplace=True)
    file.drop('zipcode', axis=1, inplace=True)
    file.drop('lat', axis=1, inplace=True)
    file.drop('long', axis=1, inplace=True)

    y = file['price']
    x = file.drop('price', axis=1)

    '''
        Use random_state param to always have the same train/test split, so the model
        score stays the same on any execution
    '''
    x_practice, x_test, y_practice, y_test = train_test_split(x, y, test_size=0.3)

    model = LinearRegression()
    model.fit(x_practice, y_practice)

    print(model.score(x_test, y_test))


if __name__ == '__main__':
    os.chdir('..')
    main()
