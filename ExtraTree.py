import numpy as np
import domain


def constant_attributes(inputs):
    compare = inputs[0]
    for other in inputs:
        if np.array_equal(compare, other) is False:
            return False

    return True


def constant_output(outputs):
    compare = outputs[0]
    for other in outputs:
        if compare != other:
            return False

    return True


def generate_splits(inputs):
    """
    Generates the splits for each attributes
    """
    splits = []
    inputs = np.array(inputs)
    for i in range(np.shape(inputs)[1]):
        attributes = inputs[:, i]

        s = round(np.random.uniform(np.min(attributes), np.max(attributes)), 2)

        splits.append(s)

    return splits


def score(s, index, inputs, outputs):
    left_subset = []
    right_subset = []

    for x, y in zip(inputs, outputs):
        if x[index] < s:
            left_subset.append(y)
        else:
            right_subset.append(y)

    var_y_s = np.var(outputs)
    var_y_Sl = np.var(left_subset)
    var_y_Sr = np.var(right_subset)

    ratio_Sl = len(left_subset)/len(outputs)
    ratio_Sr = len(right_subset)/len(outputs)

    numerator = var_y_s - ratio_Sl*var_y_Sl - ratio_Sr*var_y_Sr

    return numerator/var_y_s


def split_maximize(splits, inputs, outputs):
    """
    Return the index of the split that has a maximum score
    """
    scores = []
    for index, s in enumerate(splits):
        split_score = score(s, index, inputs, outputs)
        scores.append(split_score)

    split_index = np.argmax(scores)
    max_split = round(splits[split_index], 2)
    return max_split, split_index


class Tree:
    """
    Extra-Tree class
    """
    def __init__(self, inputs, outputs, n_min=5):
        """
        Generates an Extra-Tree using the pairs input/output
        """
        self.isLeaf = False
        self.value = None

        self.attribute_index = None
        self.cut_point = None
        self.left_subtree = None
        self.right_subtree = None

        if len(inputs) == 0 or len(inputs) != len(outputs):
            print('Set is empty or input set not same size of output set !')
        elif len(inputs) < n_min or constant_attributes(inputs) or constant_output(outputs):
            self.isLeaf = True
            self.value = np.mean(outputs)
        else:
            splits = generate_splits(inputs)
            self.cut_point, self.attribute_index = split_maximize(splits, inputs, outputs)

            left_inputs = []
            left_outputs = []
            right_inputs = []
            right_outputs = []

            for x, y in zip(inputs, outputs):
                if x[self.attribute_index] < self.cut_point:
                    left_inputs.append(x)
                    left_outputs.append(y)
                else:
                    right_inputs.append(x)
                    right_outputs.append(y)

            self.left_subtree = Tree(left_inputs, left_outputs, n_min)
            self.right_subtree = Tree(right_inputs, right_outputs, n_min)

    def __call__(self, x):
        if self.isLeaf is True:
            return self.value
        else:
            if x[self.attribute_index] < self.cut_point:
                return self.left_subtree(x)
            else:
                return self.right_subtree(x)


if __name__ == '__main__':
    i = [[60.45, -3.3, 4.3], [15.1, -2.0, -3.2], [4.23, 13.3, 6.0], [-7.2, -1.3, 9.3]]
    o = [4, 3, 5, 8]
    t = Tree(i, o, 2)

    print(t([3, 4, 5]))
