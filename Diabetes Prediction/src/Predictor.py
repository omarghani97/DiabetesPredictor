import injestData
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    # d = injestData.Dog()
    # d.Bark()
    # print(type(d))

    dataset = pd.read_csv('diabetes.csv')
    print(dataset.head())
    print(len(dataset))
    print(dataset.columns.tolist())  # displays all columns

    no_zeros = ['Glucose', 'Insulin', 'BloodPressure', 'SkinThickness', 'BMI', 'Age']  # created a list of column headings where zero cannot be accepted

    # create a for loop to scan those columns and replace 0's with mean values

    for column in no_zeros:
        dataset[column] = dataset[column].replace(0, np.NaN)
        mean = int(dataset[column].mean(skipna=True))
        dataset[column] = dataset[column].replace(np.NaN, mean)

