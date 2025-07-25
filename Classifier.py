from collections import defaultdict
import utils, pandas as pd, import numpy as np

def separateByClass(df):
    ret = defaultdict(list)
    for i, row in df.iterrows():
        class_name = row['Class']
        sepal_length = row['Sepal length (cm)']
        sepal_width = row['Sepal width (cm)']
        petal_length = row['Petal length (cm)']
        petal_width = row['Petal width (cm)']

        ret[class_name].append((sepal_length, sepal_width, petal_length, petal_width))
    return ret

def printDataSepClass(dataSepClass):
    for class_name, val_list in dataSepClass.items():
        print(f"Class: {class_name}")
        for val in val_list:
            print(val)
    
def readData():
    dataFrame = pd.read_csv('iris.csv', sep=',', header=0, encoding='utf-8')
    return dataFrame

def main():
    dataFrame = readData()
    dataSepClass = separateByClass(dataFrame)


if __name__ == '__main__':
    main()