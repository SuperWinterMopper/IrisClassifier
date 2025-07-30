from collections import defaultdict
from typing import Dict, List, Tuple
import utils, pandas as pd, numpy as np

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

def summarizeData(data: Dict[str, List[Tuple[float, float, float, float]]]) -> Dict[str, List[Tuple[float, float, int]]]: 
    summaries = {}
    for class_name, vals in data.items():
        summaries[class_name] = [(utils.mean(col), utils.std(col), len(col)) for col in zip(*vals)]
    return summaries

def printDataSepClass(dataSepClass):
    for class_name, val_list in dataSepClass.items():
        print(f"Class: {class_name}")
        for val in val_list:
            print(val)
    
def readData():
    dataFrame = pd.read_csv('iris.csv', sep=',', header=0, encoding='utf-8')
    return dataFrame

def classP(summaries: Dict[str, List[Tuple[float, float, int]]], row: Tuple[float, float, float, float]) -> Dict[str, float]:
    probabilities = defaultdict(float)
    total_rows = sum([data[0][2] for data in summaries.values()])

    for class_name, vals in summaries.items():
        p = summaries[class_name][0][2] / float(total_rows)
        for i in range(len(vals)):
            u, std, _ = vals[i]
            p *= utils.gaussianPDF(row[i], u, std)
        probabilities[class_name] = p
    return probabilities

def predict(summaries, instance):
    probabilities = classP(summaries, instance)
    best_label, best_prob = None, -1
    for class_name, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_name
    return best_label

def evaluate_model():
    df = readData()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    split_point = len(df) // 2
    training_set = df.iloc[:split_point]
    testing_set = df.iloc[split_point:]
    
    dataSepClass = separateByClass(training_set)
    model = summarizeData(dataSepClass)
    
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))
    
    for i, row in testing_set.iterrows():
        features = (row['Sepal length (cm)'], row['Sepal width (cm)'], row['Petal length (cm)'], row['Petal width (cm)'])
        actual = row['Class']
        predicted = predict(model, features)
        
        if actual == predicted:
            correct += 1
            
        confusion[actual][predicted] += 1
    
    accuracy = (correct / len(testing_set)) * 100
    
    print(f"Model Accuracy: {accuracy:.2f}%")
    print("\nConfusion Matrix:")
    
    classes = sorted(set(df['Class']))
    print(f"{'Actual/Predicted':20}", end="")
    for cls in classes:
        print(f"{cls.split('-')[1]:15}", end="")
    print()
    
    for actual in classes:
        print(f"{actual.split('-')[1]:20}", end="")
        for predicted in classes:
            print(f"{confusion[actual][predicted]:15}", end="")
        print()
    
    return model, accuracy

def main():
    evaluate_model()

if __name__ == '__main__':
    main()