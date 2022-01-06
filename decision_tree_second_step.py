import copy
import sys
import math
import random
import csv
sys.setrecursionlimit(10 ** 8)


class TreeNode:
    def __init__(self, parent_node, left_child_node, right_sibling_node, value_that_has, attrib_test):
        self.parent_node = parent_node
        self.left_child = left_child_node
        self.right_sibling = right_sibling_node
        self.value_that_has = value_that_has
        self.attrib_test = attrib_test
        self.output = None
        self.p = None
        self.n = None

def plurality_value(examples_isolated):
    output_numbers = {False: 0, True: 0}
    for data in examples_isolated:
        output_numbers[data[1]] += 1
    if output_numbers[False] > output_numbers[True]:
        return False, output_numbers
    elif output_numbers[False] < output_numbers[True]:
        return True, output_numbers
    else:
        return bool(random.getrandbits(1)), output_numbers


def classified_property(example_param):
    same_classification_1 = example_param[0][1]
    for i in range(1, len(example_param)):
        same_classification_1 = same_classification_1 and example_param[i][1]
    same_classification_2 = example_param[0][1]
    for i in range(1, len(example_param)):
        same_classification_2 = same_classification_2 or example_param[i][1]
    return not same_classification_2 or same_classification_1, same_classification_1


def learn_decision_tree(node, example_param, parent_example, attributes_local):
    if not example_param:
        the_tuple = plurality_value(parent_example)
        node.output = the_tuple[0]
        node.p = the_tuple[1][True]
        node.n = the_tuple[1][False]
        return node
    the_tuple_2 = classified_property(example_param)
    if the_tuple_2[0]:
        node.output = the_tuple_2[1]
        if node.output:
            node.p = len(example_param)
        else:
            node.n = len(example_param)
        return node
    elif not attributes_local:
        the_tuple = plurality_value(example_param)
        node.output = the_tuple[0]
        node.p = the_tuple[1][True]
        node.n = the_tuple[1][False]
        return node
    else:
        A = importance(attributes_local, example_param)
        node.attrib_test = A
        a_set = set()
        a_set.add(A)
        attributes_local.difference_update(a_set)
        flag_1 = True
        left_sibling = None
        for value in attributes_domain[A]:
            exs = []
            for data in example_param:
                if data[0][attributes_index[A]] == value:
                    exs.append(data)
            subtree = learn_decision_tree(TreeNode(node, None, None, value, None), exs, example_param,
                                          attributes_local)
            if subtree == node:
                subtree = copy.deepcopy(subtree)
                subtree.left_child = None
                subtree.value_that_has = value
                subtree.attrib_test = None
            if flag_1:
                left_sibling = subtree
                flag_1 = False
                node.left_child = subtree
            else:
                left_sibling.right_sibling = subtree
                left_sibling = subtree
        return node


def show(tree, length, depth):
    if tree is None:
        return
    else:
        print("The most common output is: " + str(tree.output), end='\n' + ('|' + ' ' * length) * depth)
        print("Number of True: " + str(tree.p), end='\n' + ('|' + ' ' * length) * depth)
        print("Number of False: " + str(tree.n), end='\n' + ('|' + ' ' * length) * depth)
        print("The attribute that is tested: " + str(tree.attrib_test), end='\n' + ('|' + ' ' * length) * depth)
        print("The value that is gotten : " + str(tree.value_that_has), end='')
        x = tree.left_child
        if x is not None:
            print(('\n' + ('|' + ' ' * length) * depth + '|') * (10 * 1) + "-" * (
                        20 * 1), end="")
        while x is not None:
            show(x, 20 * 1, depth + 1)
            copy_of = x
            x = x.right_sibling
            if x is not None:
                print(('\n' + ('|' + ' ' * length) * depth + '|') * (10 * 1) + "-" * (
                            20 * 1), end="")
        return


def classify(tree, instance):
    if tree.left_child is None:
        return tree.output
    the_index = attributes_index[tree.attrib_test]
    the_value = instance[the_index]
    x = tree.left_child
    while x.value_that_has != the_value:
        x = x.right_sibling
    return classify(x, instance)




def importance(attributes_local, examples_local):
    gain_per_attrib = dict(zip(attributes_local, [0] * len(attributes_local)))
    for attribute in attributes_local:
        set_of_values = attributes_domain[attribute]
        remainder_attribute = 0
        for value in set_of_values:
            dict_of_of_classify = {True: 0, False: 0}
            for data in examples_local:
                if data[0][attributes_index[attribute]] == value:
                    dict_of_of_classify[data[1]] += 1
            if dict_of_of_classify[True] == 0 and dict_of_of_classify[False] == 0:
                remainder_attribute += 0
            else:
                probability = dict_of_of_classify[True] / (dict_of_of_classify[True] + dict_of_of_classify[False])
                if probability == 0 or probability == 1:
                    remainder_attribute += 0
                else:
                    remainder_attribute += ((dict_of_of_classify[True] + dict_of_of_classify[False]) / total_examples) * (-1) *\
                                    ((probability * math.log2(probability)) + ((1 - probability) * math.log2(1 - probability)))
        gain_per_attrib[attribute] = total_entropy - remainder_attribute
    maximum = max(gain_per_attrib.values())
    list_of_candidate_attribs = []
    for attribute in gain_per_attrib.keys():
        if gain_per_attrib[attribute] == maximum:
            list_of_candidate_attribs.append(attribute)
    return random.choice(list_of_candidate_attribs)


attributes_index = {"Pregnancies": 0, "Glucose": 1, "BloodPressure": 2, "SkinThickness": 3, "Insulin": 4,
                    "BMI": 5, "DiabetesPedigreeFunction": 6, "Age": 7}
with open("diabetes.csv", mode='r') as file:
    reader = csv.reader(file)
    dataset_1 = []
    dataset_2 = []
    for line in reader:
        dataset_1.append(line)
    dataset_1.pop(0)
    for i in range(len(dataset_1)):
        example = []
        length = len(dataset_1[i])
        for j in range(length - 1):
            example.append(float(dataset_1[i][j]))
        dataset_2.append((example, bool(int(dataset_1[i][length - 1]))))
    del dataset_1
attributes_domain = dict()
for attribute in set(attributes_index.keys()):
    index = attributes_index[attribute]
    list_of_values = []
    simple_set = set()
    for example in dataset_2:
        list_of_values.append(example[0][index])
        simple_set.add(example[0][index])
    attributes_domain[attribute] = simple_set
    if attribute == "BMI" or attribute == "DiabetesPedigreeFunction":
        set_of_values = set()
        maximum = round(max(list_of_values), 3)
        minimum = round(min(list_of_values), 3)
        the_length = 90
        length_of_intervals = (maximum - minimum) / the_length
        minimum_1 = ((-1) * math.inf, minimum)
        maximum_1 = (maximum, math.inf)
        cum_sum = 0
        for i in range(the_length):
            set_of_values.add((minimum, round(minimum + length_of_intervals, 5)))
            minimum = round(minimum + length_of_intervals, 3)
        set_of_values.add(minimum_1)
        set_of_values.add(maximum_1)
        attributes_domain[attribute] = set_of_values
        for element in dataset_2:
            for values in set_of_values:
                if values[0] <= element[0][index] < values[1]:
                    element[0][index] = values
                    break
train = random.choices(dataset_2, k=math.floor(0.8 * len(dataset_2)))
test = []
for element in dataset_2:
    if element not in train:
        test.append(element)
root = TreeNode(None, None, None, None, None)
x = 0
y = 0
total_examples = len(train)
for element in train:
    if element[1]:
        x += 1
    else:
        y += 1
probability_of_pos = x / (x + y)
total_entropy = (-1) * (probability_of_pos * math.log2(probability_of_pos) + (1 - probability_of_pos) * math.log2(
    1 - probability_of_pos))
model = learn_decision_tree(root, train, None, set(attributes_index.keys()))
correct = 0
not_correct = 0
show(model, 0, 0)
print()
for test_element in test:
    the_predicted = classify(model, test_element[0])
    if the_predicted == test_element[1]:
        correct += 1
    else:
        not_correct += 1
    print("The Actual class is", test_element[1], "The predicted is", the_predicted)
print()
print("The precision is", str(round((correct / (not_correct + correct)) * 100)) + '%')
