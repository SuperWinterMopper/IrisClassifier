from collections import defaultdict
import pandas as pd

def separateByClass(df):
    ret = defaultdict(list)
    for i, row in df.iterrows():
        class_name = row['Class']
        sepal_length = row['Sepal length (cm)']
        sepal_width = row['Sepal width (cm)']
        petal_length = row['Petal length (cm)']
        petal_width = row['Petal width (cm)']

def readData():
    dataFrame = pd.read_csv('iris.csv', sep=',', header=0, encoding='utf-8')
    return dataFrame

def main():
    dataFrame = readData()

if __name__ == '__main__':
    main()