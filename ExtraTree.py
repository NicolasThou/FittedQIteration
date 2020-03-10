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
    for i in range(inputs.shape[1]):
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
    max = []
    for index, s in enumerate(splits):
        split_score = score(s, index, inputs, outputs)
        max.append(split_score)

    return np.max(max), np.argmax(max)


class Tree:
    """
    Extra-Tree class
    """
    def __init__(self, inputs, outputs, n_min):
        """
        Generates a Extra-Tree using the pairs input/output
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
            right_intputs = []
            right_outputs = []

            for x, y in zip(inputs, outputs):
                if x[self.attribute_index] < self.cut_point:
                    left_inputs.append(x)
                    left_outputs.append(y)
                else:
                    right_intputs.append(x)
                    right_outputs.append(y)

            self.left_subtree = Tree(left_inputs, left_outputs, n_min)
            self.right_subtree = Tree(right_intputs, right_outputs, n_min)

    def __call__(self, x):
        if self.isLeaf is True:
            return self.value
        else:
            if x[self.attribute_index] < self.cut_point:
                return self.left_subtree.forward(x)
            else:
                return self.right_subtree.forward(x)


if __name__ == '__main__':
    t = Tree([[1, 2, 3]], [1.2], 2)
    print(t([3, 4, 5]))
