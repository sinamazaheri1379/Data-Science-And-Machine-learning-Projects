import numpy as np
import binarytree as bt
import sys
import datetime
from binarytree import Node
sys.setrecursionlimit(10 ** 8)
np.seterr(all='ignore')


def solve(root, x):
    if root is not None:
        if root.val == float(-3):
            return x
        elif root.val == float(-1):
            return np.around(np.e, 5)
        elif root.val == float(-2):
            return np.round(np.pi, 5)
        elif root.val not in list(map(float, range(7))):
            return np.around(root.val, 5)
        else:
            A = solve(root.left, x)
            B = solve(root.right, x)
            if root.val == float(0):
                return np.around(np.add(A, B), 5)
            elif root.val == float(1):
                return np.around(np.subtract(A, B), 5)
            elif root.val == float(2):
                return np.around(np.multiply(A, B), 5)
            elif root.val == float(3):
                if A == 0 and B == 0:
                    return float(1)
                if A != 0 and B == 0:
                    return float("inf")
                return np.around(np.divide(A, B), 5)
            elif root.val == float(4):
                if B < 0:
                    B = np.multiply(-1, B)
                    A = 1 / A
                return np.around(np.power(A, B), 5)
            elif root.val == float(5):
                return np.around(np.sin(np.multiply(A, np.pi / 180)), 5)
            else:
                return np.around(np.cos(np.multiply(A, np.pi / 180)), 5)
    else:
        return float(0)


def print_formula(root, formula):
    if root is not None:
        if root.val == float(-3):
            return formula + str('x')
        elif root.val == float(-1):
            return formula + str('exp')
        elif root.val == float(-2):
            return formula + str('pi')
        elif root.val not in list(map(float, range(7))):
            return formula + str(root.val)
        else:
            A = print_formula(root.left, formula)
            B = print_formula(root.right, formula)
            if root.val == float(0):
                return formula + '(' + A + '+' + B + ')'
            elif root.val == float(1):
                return formula + '(' + A + '-' + B + ')'
            elif root.val == float(2):
                return formula + '(' + A + '*' + B + ')'
            elif root.val == float(3):
                return formula + '(' + A + '/' + B + ')'
            elif root.val == float(4):
                return formula + '(' + A + '^' + B + ')'
            elif root.val == float(5):
                return formula + 'Sin' + '(' + A + ')'
            else:
                return formula + 'Cos' + '(' + A + ')'
    else:
        return formula + ""


def err_computation(tree, points):
    list_of_values = []
    for x, y in zip(list(points[0]), list(points[1])):
        y_bar = np.around(solve(tree, x), 2)
        if not np.isfinite(y_bar):
            continue
        square_inverse = np.around(np.power(np.subtract(y_bar, y), 2), 5)
        if not np.isfinite(square_inverse):
            continue
        list_of_values.append(square_inverse)
    return np.around(1 / np.sqrt(np.mean(np.array(list_of_values))), 5)


def weighted_by(population, points):
    fitness_respect_to_tree = dict()
    for tree in population:
        fitness = err_computation(tree, points)
        if not np.isfinite(fitness):
            continue
        fitness_respect_to_tree[tree] = fitness
    return fitness_respect_to_tree


def weighted_random_choice(weights_local, j):
    all_fitness = np.array(list(weights_local.values()))
    finite_trees = list(weights_local.keys())
    all_fitness_rounded = np.around(all_fitness, 2)
    fitness_probability = np.divide(all_fitness_rounded, np.sum(all_fitness_rounded))
    # while not np.isfinite(fitness_probability).all():
    #     print("I'm in loop")
    #     tree_indexes_that_is_finite = list(np.where(np.any(np.isfinite(fitness_probability))))
    #     finite_trees = []
    #     for i in range(len(weights_local.keys())):
    #         if i in tree_indexes_that_is_finite:
    #             finite_trees.append(tree_list[i])
    #     all_fitness = []
    #     for element in list(weights_local.keys()):
    #         if element in finite_trees:
    #             all_fitness.append(weights_local[element])
    #     all_fitness_rounded = np.round(all_fitness, 2)
    #     fitness_probability = np.divide(all_fitness_rounded, np.sum(all_fitness_rounded))
    indexes = np.random.choice(np.arange(len(all_fitness)),  2, replace=True, p=fitness_probability)
    return (finite_trees[indexes[0]], finite_trees[indexes[1]])





def reproduce(tree_1, tree_2):
    set_result_1 = list(set(tree_1.levelorder).difference(set(tree_1.leaves)))
    chosen_node_tree_1_offset = np.random.choice(np.arange(len(set_result_1)), 1)[0]
    chosen_node_tree_1 = set_result_1[chosen_node_tree_1_offset]
    way_1 = True
    if chosen_node_tree_1.right is not None:
        way_1 = bool(np.random.randint(0, 2))
        if way_1:
            temp_node_1 = chosen_node_tree_1.left
        else:
            temp_node_1 = chosen_node_tree_1.right
    else:
        temp_node_1 = chosen_node_tree_1.left
    set_result_2 = list(set(tree_2.levelorder).difference(set(tree_2.leaves)))
    chosen_node_tree_2_offset = np.random.choice(np.arange(len(set_result_2)), 1)[0]
    chosen_node_tree_2 = set_result_2[chosen_node_tree_2_offset]
    way_2 = True
    if chosen_node_tree_2.right is not None:
        way_2 = bool(np.random.randint(0, 2))
        if way_2:
            temp_node_2 = chosen_node_tree_2.left
        else:
            temp_node_2 = chosen_node_tree_2.right
    else:
        temp_node_2 = chosen_node_tree_2.left
    if way_2:
        chosen_node_tree_2.left = temp_node_1
    else:
        chosen_node_tree_2.right = temp_node_1
    if way_1:
        chosen_node_tree_1.left = temp_node_2
    else:
        chosen_node_tree_1.right = temp_node_2
    randomness = bool(np.random.randint(0, 2))
    if randomness:
        return tree_1
    else:
        return tree_2


def mutate(child_1_old, mutation_rate, max_height):
    probability = [1 - mutation_rate, mutation_rate]
    respect_child_1 = []
    for node in child_1_old.levelorder:
        Flag = bool(np.random.choice([0, 1], 1, p=probability)[0])
        if Flag:
            i = np.random.randint(1, max_height, 1)[0]
            list_length = np.power(2, i + 1) - 1
            left = int(np.ceil((list_length - 1) / 2))
            variable = [-3] * (list_length - 1)
            random_tree = produce_random_tree(list_length, left, variable)
            respect_child_1.append((node, bt.build(list(random_tree))))
    for old_node, mutated_node in respect_child_1:
        old_node.left = mutated_node
    return child_1_old


def produce_random_tree(list_length, left, variable):
    random_ops = np.random.choice(np.arange(7), left)
    random_operands = np.round(np.random.normal(np.mean(initial_points[1]), np.std(initial_points[1]), list_length - 1),
                               4)
    random_const = np.random.choice(list_of_const, list_length - 1)
    chooser_rand = [0, 0, 0, 0]
    chooser = [random_ops, variable, random_operands, random_const]
    tree_list = np.array([None] * list_length)
    array = chooser[0]
    selected = float(array[chooser_rand[0]])
    tree_list[0] = selected
    chooser_rand[0] += 1
    tree_list[1] = float("inf")
    if selected == 5 or selected == 6:
        if 2 < list_length:
            tree_list[2] = None
    else:
        tree_list[2] = float("inf")
    for j in range(1, left):
        if tree_list[j] == float("inf"):
            rand = np.random.choice([0, 1, 2, 3], 1, p=[0.55, 0.43, 0.01, 0.01])[0]
            array = chooser[rand]
            selected = array[chooser_rand[rand]]
            chooser_rand[rand] += 1
            tree_list[j] = float(selected)
            if rand == 0:
                tree_list[2 * j + 1] = float("inf")
                if selected == 6 or selected == 5:
                    if 2 * j + 2 < list_length:
                        tree_list[2 * j + 2] = None
                else:
                    tree_list[2 * j + 2] = float("inf")
    for j in range(left, list_length):
        if tree_list[j] == float("inf"):
            rand = np.random.choice([1, 2, 3], 1, p=[0.5, 0.4, 0.1])[0]
            array = chooser[rand]
            selected = array[chooser_rand[rand]]
            chooser_rand[rand] += 1
            tree_list[j] = float(selected)
    return tree_list



before_1 = np.array([-4, -5, -6, -7, -8, -9, -10, -11, -12, -13])
after_1 = (-1) * before_1 - 1
before_2 = np.array([2, 3, 5, 6 ,34, 234, 13, 3 ,2123, 42, 1, 3, 21, 32, 13, 1])
after_2 = np.power(before_2, 2)
initial_points = (np.hstack((before_1, before_2)), np.hstack((after_1, after_2)))
population_1 = set()
list_of_keys = np.array(['+', '-', '*', '/', '^', "Sin", "Cos"])
dict_of_const = {-1: 'exp', -2: 'pi'}
list_of_const = np.array(list(dict_of_const.keys()))
max_height = 10
number_per_each = 400

for i in range(1, max_height):
    list_length = np.power(2, i + 1) - 1
    left = int(np.ceil((list_length - 1) / 2))
    right = list_length - left
    variable = [-3] * (list_length - 1)
    for k in range(number_per_each):
        tree_list = produce_random_tree(list_length, left, variable)
        population_1.add(bt.build(list(tree_list)))

j = 0
the_best_trees = []
begin = datetime.datetime.now()
while len(population_1) != 0:
    weights = weighted_by(population_1, initial_points)
    if len(weights.keys()) != 0:
        the_max_value = max(list(weights.values()))
        for element in list(weights.keys()):
            if weights[element] == the_max_value:
                the_best_trees.append((element, the_max_value))
    else:
        break
    population_2 = list()
    list_of_kies = list(weights.keys())
    for i in range(len(list_of_kies)):
        parent_tuple = weighted_random_choice(weights, j)
        parent_1 = parent_tuple[0]
        parent_2 = parent_tuple[1]
        child = reproduce(parent_1, parent_2)
        child_new = mutate(child, np.divide(1, len(population_1)), max_height)
        population_2.append(child_new)
    population_1 = population_2
    j += 1
end = datetime.datetime.now()
for i in range(len(the_best_trees)):
    print(the_best_trees[i][0], the_best_trees[i][1], j,  end - begin, sep="\n\n")
    print("Expression : " + print_formula(the_best_trees[i][0], ''))
