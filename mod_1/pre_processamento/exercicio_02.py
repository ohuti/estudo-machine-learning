import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    """
        Esse excercício tem como objetivo calcular a correlação entre variáveis do dataframe e gerar um gráfico de
        heatmap.
    """

    data = pd.read_csv('source_files/2015-building-energy-benchmarking.csv')
    pd.set_option('display.max_columns', len(data.columns))

    print(data.corr(method='pearson'))
    plt.figure(figsize=(20, 20))
    sns.heatmap(data.corr(method='pearson'))
    plt.show()

    return


if __name__ == '__main__':
    os.chdir('.')
    main()
