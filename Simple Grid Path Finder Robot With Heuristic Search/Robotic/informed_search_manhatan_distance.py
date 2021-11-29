import sys
import lxml.etree
import heapq
from math import fabs


class Node:
    def __init__(self, state, parent_node, path_cost, action):
        self.state = state
        self.parent_node = parent_node
        self.path_cost = path_cost
        self.action = action

    def get_state(self):
        return self.state

    def get_path_cost(self):
        return self.path_cost

    def get_parent_node(self):
        return self.parent_node


def expand(node, plan):
    state = node.get_state()
    all_children = set()
    for single_action in action(state, plan):
        the_tuple = result(node, single_action)
        cost = node.get_path_cost() + 1
        all_children.add(Node(the_tuple, node, cost, single_action))
    return all_children


def result(node, action):
    state = node.get_state()
    the_tuple = [state[0], state[1]]
    if action == 'L':
        the_tuple[1] -= 1
    elif action == 'R':
        the_tuple[1] += 1
    elif action == 'U':
        the_tuple[0] -= 1
    else:
        the_tuple[0] += 1
    the_tuple.append("empty")
    return tuple(the_tuple)


def action(state, plan):
    all_actions = {'D', 'U', 'L', 'R'}
    state_y = state[0]
    state_x = state[1]
    length_y = len(plan)
    length_x = len(plan[0])
    if state_y + 1 == length_y or plan[state_y + 1][state_x].text == "obstacle":
        all_actions.difference_update({'D'})
    if state_y - 1 == -1 or plan[state_y - 1][state_x].text == "obstacle":
        all_actions.difference_update({'U'})
    if state_x + 1 == length_x or plan[state_y][state_x + 1].text == "obstacle":
        all_actions.difference_update({'R'})
    if state_x - 1 == -1 or plan[state_y][state_x - 1].text == "obstacle":
        all_actions.difference_update({'L'})
    return all_actions


def evaluate_function(node, goal):
    node_state = node.get_state()
    return fabs(node_state[0] - goal[0]) + fabs(node_state[1] - goal[1]) + node.get_path_cost()





tree = lxml.etree.parse(sys.argv[1])
a_1 = tree.xpath('//cell[text()="robot"]')[0]
a_2 = a_1.getparent()
plan = a_2.getparent()
initial_state = (plan.index(a_2), a_2.index(a_1), "robot")
robot_node = Node(initial_state, None, 0, None)

a_1 = plan.xpath('//cell[text()="Battery"]')[0]
a_2 = a_1.getparent()
plan = a_2.getparent()
goal_tuple = (plan.index(a_2), a_2.index(a_1))
frontier = []
reached = {}
index = 0
heapq.heappush(frontier, (evaluate_function(robot_node, goal_tuple), index, robot_node))
reached[(initial_state[0], initial_state[1])] = robot_node
number_of_steps = 0
failure_or_success = False
while frontier:
    node_tuple = heapq.heappop(frontier)
    node = node_tuple[2]
    if (node.get_state()[0], node.get_state()[1]) == goal_tuple:
        failure_or_success = True
        break
    for child in expand(node, plan):
        current_state = child.get_state()
        if (current_state[0], current_state[1]) not in reached.keys() or child.get_path_cost() < reached[(current_state[0], current_state[1])].get_path_cost():
            reached[(current_state[0], current_state[1])] = child
            index += 1
            heapq.heappush(frontier, (evaluate_function(child, goal_tuple), index, child))
if failure_or_success:
    print("I've found an optimal path!!!\n")
    path_node = node
    stack = []
    while path_node is not None:
        stack.append(path_node)
        path_node = path_node.get_parent_node()
    number_of_steps = len(stack)
    step = 0
    while stack:
        single_state = stack.pop()
        coordinate = single_state.get_state()
        print("Step: " + str(step) + "\n")
        string = ''
        for i in range(len(plan)):
            string = ''
            for j in range(len(plan[0])):
                string += '|'
                if (i, j) == (coordinate[0], coordinate[1]):
                    string += 'R'
                elif plan[i][j].text == "Battery":
                    string += 'B'
                elif plan[i][j].text == "obstacle":
                    string += '*'
                else:
                    string += ' '
            string += '|'
            print(string)
        print("\n\n\n")
        step += 1
else:
    print("No Way has been found!!")










