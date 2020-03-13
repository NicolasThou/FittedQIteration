import numpy as np
import domain


def constant_attributes(inputs):
    """
    Check wether an attribute is constant over all the inputs
    """
    inputs = np.array(inputs)

    # check for all attributes
    for i in range(np.shape(inputs)[1]):
        attributes = inputs[:, i]
        # check if an attribute is constant
        if np.all(attributes == attributes[0]):
            return True

    return False


def constant_output(outputs):
    """
    Check wether all the outputs are constant in a set
    """
    outputs = np.array(outputs)

    # compare each element with the first
    if np.all(outputs == outputs[0]):
        return True

    return False


def generate_splits(inputs):
    """
    Generates random splits for each attributes
    """
    splits = []
    inputs = np.array(inputs)

    # for each attribute of the inputs compute a random split
    for i in range(np.shape(inputs)[1]):
        attributes = inputs[:, i]

        s = round(np.random.uniform(np.min(attributes), np.max(attributes)), 2)

        splits.append(s)

    return splits


def score(s, index, inputs, outputs):
    """
    Return the score of a split
    """
    left_subset = []
    right_subset = []

    # for each input.output pair, build the subtrees set
    for x, y in zip(inputs, outputs):

        if x[index] < s:
            left_subset.append(y)
        else:
            right_subset.append(y)

    # computes the components of the score of the split according the formula of the paper
    # see : " P. Geurts, D. Ernst and L. Wehenkel : Extremely Randomized Trees"
    var_y_s = np.var(outputs)
    var_y_Sl = np.var(left_subset)
    var_y_Sr = np.var(right_subset)
    ratio_Sl = len(left_subset)/len(outputs)
    ratio_Sr = len(right_subset)/len(outputs)
    numerator = var_y_s - ratio_Sl*var_y_Sl - ratio_Sr*var_y_Sr

    return numerator/var_y_s


def split_maximize(splits, inputs, outputs):
    """
    Return the value and the index of the split that has a maximum score
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
            # if one stopping condition is true, declare the Tree as a leaf
            # and associates its the mean value of the outputs
            self.isLeaf = True
            self.value = np.mean(outputs)
        else:
            # generates all the random splits
            splits = generate_splits(inputs)

            # choose the split with the maximum score
            self.cut_point, self.attribute_index = split_maximize(splits, inputs, outputs)

            left_inputs = []
            left_outputs = []
            right_inputs = []
            right_outputs = []

            # build the sets for growing the subtrees
            for x, y in zip(inputs, outputs):
                if x[self.attribute_index] < self.cut_point:
                    left_inputs.append(x)
                    left_outputs.append(y)
                else:
                    right_inputs.append(x)
                    right_outputs.append(y)

            # build the subtrees
            self.left_subtree = Tree(left_inputs, left_outputs, n_min)
            self.right_subtree = Tree(right_inputs, right_outputs, n_min)

    def __call__(self, X):
        """
        Predict the output for X
        """
        if self.isLeaf is True:
            # if the Tree is just a leaf, return its value
            return self.value
        else:
            # else choose a subtree according to the cut-point
            if X[self.attribute_index] < self.cut_point:
                return self.left_subtree(X)
            else:
                return self.right_subtree(X)


def predict(tree, X):
    Y = []
    for sample in X:
        Y.append(tree(sample))

    return np.round(Y, 2)


if __name__ == '__main__':
    X = [[2, 3], [4, 5], [2, 5]]
    Y = [2, 5, 4]

    t = Tree(X, Y, 2)

    x = [[4, 6], [2, 7]]
    print(predict(t, x))
