import numpy as np
import re


def recursive_split(x, y):
    if is_pure(y):
        return most_common(y)

    gains_by_predictors = np.array([calc_info_gain(x_predictor, y) for x_predictor in x.T])
    index_of_best_attr_for_split = np.argmax(gains_by_predictors)

    if gains_by_predictors[index_of_best_attr_for_split] < 1e-6:
        return most_common(y)

    result = {}
    possible_values = np.unique(x[:, index_of_best_attr_for_split])
    for attr_val in possible_values:
        indexes_for_subset = np.where(x[:, index_of_best_attr_for_split] == attr_val)
        x_subset = x[indexes_for_subset]
        y_subset = y[indexes_for_subset]
        node_key = form_node_key(index_of_best_attr_for_split, attr_val)
        result[node_key] = recursive_split(x_subset, y_subset)
    return result


def is_pure(a):
    return len(set(a)) <= 1


def most_common(classes):
    val, counts = np.unique(classes, return_counts=True)
    index = np.argmax(counts)
    return val[index]


def calc_info_gain(x_attrs, y_values):
    overall_entropy = entropy(y_values)

    values, frequencies = get_val_frequencies(x_attrs)
    split_entropy = 0
    for val, freq in zip(values, frequencies):
        split_entropy += freq * entropy(y_values[x_attrs == val])
    return overall_entropy - split_entropy


def entropy(s):
    res = 0.0
    *temp, frequencies = get_val_frequencies(s)
    for frequency in frequencies:
        if frequency != 0.0:
            res -= frequency * np.log2(frequency)
    return res


def get_val_frequencies(s):
    val, counts = np.unique(s, return_counts=True)
    frequencies = counts.astype("float")/len(s)
    return val, frequencies


def form_node_key(attr_index, attr_val):
    return "x_{} = {}".format(attr_index, attr_val)


def print_result_tree_recursive(tree, depth=0):
    if type(tree) is not dict:
        print_with_depth_structure(tree, depth)
        return
    for val, son in tree.items():
        print_with_depth_structure(val, depth, ":")
        print_result_tree_recursive(son, depth+1)


def print_with_depth_structure(val, depth, str_end=""):
    print("{}{}{}".format("\t"*depth, val, str_end))


def predict_recursive(tree, X_vals):
    if type(tree) is not dict:
        return tree

    key_example = list(tree.keys())[0]
    attr_index = get_attr_index(key_example)
    node_key = form_node_key(attr_index, X_vals[attr_index])
    return predict_recursive(tree[node_key], X_vals)


def get_attr_index(key):
    pattern = re.compile(r"(?<=x_)\d+")
    return int(re.findall(pattern, key)[0])



