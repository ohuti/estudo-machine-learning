import os
import pandas as pd


def main():
    """
        Objetivo do exercício é demonstrar o percentual de dados faltantes de cada uma das colunas e substituir os dados
        faltantes da coluna "ENERGYSTARScore" pela mediana da mesma.
    """
    data = pd.read_csv('source_files/2015-building-energy-benchmarking.csv')
    pd.set_option('display.max_columns', len(data.columns))

    rows = data.shape[0]
    print(data.isnull().sum() / rows * 100)

    data['ENERGYSTARScore'] = data['ENERGYSTARScore'].fillna(data['ENERGYSTARScore'].median())

    print(data.isnull().sum() / rows * 100)

    return


if __name__ == '__main__':
    os.chdir('.')
    main()
