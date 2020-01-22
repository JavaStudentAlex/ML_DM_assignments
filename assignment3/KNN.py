#import libs
import pandas as pd
import numpy as np
import math

min_max_norma = None

def main():

    #prepare train data
    train_data_cols = ["color", "radius", "weight", "class"]
    train_data_types = [str, float, float, str]

    data_train = read_data("trainData.csv", train_data_cols, train_data_types)
    data_train = data_train.drop_duplicates()
    data_train = data_train.replace(0, np.nan)
    data_train = data_train.dropna()
    data_train = refactor_categorical_columns(data_train, ["color"])

    global min_max_norma
    min_max_norma = get_min_max(data_train, ["radius", "weight"])

    normalize(data_train, ["radius", "weight"])

    #prepare test data
    test_data_cols = ["color", "radius", "weight", "predicted", "class"]
    test_data_types = [str, float, float, str, str]

    test_data = read_data("testData.csv", test_data_cols, test_data_types)
    test_data.drop_duplicates()
    test_data = refactor_categorical_columns(test_data, ["color"])
    normalize(test_data, ["radius", "weight"])

    train_x = data_train.drop(["class"], axis=1).values
    train_y = data_train["class"].values
    test_x = test_data.drop(["predicted", "class"], axis=1).values

    test_data["predicted"] = predict_KNN(5, train_x, train_y, test_x)

    classes = np.unique(test_data["class"].values)

    clusses_number = len(classes)
    confusion_matrix = pd.DataFrame(index=classes, columns=classes, dtype="int32",
                                    data=np.zeros(shape=(clusses_number, clusses_number)))

    for row in test_data[["predicted", "class"]].values:
        confusion_matrix.loc[row[0], row[1]] += 1
    print(confusion_matrix)



def read_data(filename, columns, types):
    return pd.read_csv(filename, names=columns, dtype=dict(zip(columns, types)), header=0)


def refactor_categorical_columns(data, cat_columns):
    for category in cat_columns:
        new_columns = pd.get_dummies(data[category])
        data = data.drop([category], axis=1)
        data = pd.concat([new_columns, data], axis=1)
    return data

def get_min_max(data, labels):
    result = {}
    for label in labels:
        min = np.min(data[label])
        max = np.max(data[label])
        result[label] = {"min": min, "max": max}
    return result

def normalize(data, columns):
    for column in columns:
        min = min_max_norma[column]["min"]
        max = min_max_norma[column]["max"]
        data[column] = list(map(lambda x: (float(x) - min)/(max - min), data[column]))


def predict_KNN(neighbours_number, train_X, train_Y, test_X):
    predicted = []
    for measurement in test_X:
        measurement_distances = get_distances(train_X, measurement)

        #get and sort distances
        distances_classes = list(zip(measurement_distances, train_Y))
        distances_classes = np.array(sorted(distances_classes, key=lambda x: x[0]))

        nearest_classes = distances_classes[:neighbours_number][:, 1]
        un, pos = np.unique(nearest_classes, return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        predicted.append(un[maxpos])
    return predicted



def get_distances(train_neighbours, test_row):
    distances = []
    for neighbour_X in train_neighbours:
        distances.append(calc_euclidian_distance(neighbour_X, test_row))
    return distances


def calc_euclidian_distance(train_raw, test_raw):
    summ = 0.0
    for i in range(0, len(test_raw)):
        test_X_predictor = test_raw[i]
        train_X_predictor = train_raw[i]
        summ += pow(test_X_predictor - train_X_predictor, 2)
    return math.sqrt(summ)






main()