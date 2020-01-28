import numpy as np
import decision_tree as dt
import pandas as pd

x1 = ['high', 'high', 'low', 'low', 'low', 'high']
x2 = ['partly cloudy', 'sunny', 'sunny', 'cloudy', 'cloudy', 'cloudy']
X = np.array([x1, x2]).T
Y = np.array([False, False, True, False, False, True])
res = dt.recursive_split(X, Y)
dt.print_result_tree_recursive(res)

x1_1 = ['high', 'low', 'low', 'high', 'low', 'high', 'high', 'low', 'low', 'high', 'low', 'low']
x2_1 = ['sunny', 'sunny', 'cloudy', 'cloudy', 'partly cloudy', 'cloudy', 'partly cloudy', 'cloudy', 'sunny', 'cloudy', 'cloudy', 'partly cloudy']
X_1 = np.array([x1_1, x2_1]).T
Y_1 = np.array([False, True, True, False, False, True, False, True, True, False, True, True]) # ground-truth of classification

Y_1_predict = np.array([dt.predict_recursive(res, x_val) for x_val in X_1])

confusion_matrix_columns_rows = [True, False]
size = (len(confusion_matrix_columns_rows), len(confusion_matrix_columns_rows))

#predicted rows really columns
confusion_matrix = pd.DataFrame(data=np.zeros(shape=size, dtype="int32"), index=confusion_matrix_columns_rows,
                                columns=confusion_matrix_columns_rows)

for predicted_class, must_be_class in zip(Y_1_predict, Y_1):
    confusion_matrix.loc[predicted_class, must_be_class] += 1

print(confusion_matrix)




