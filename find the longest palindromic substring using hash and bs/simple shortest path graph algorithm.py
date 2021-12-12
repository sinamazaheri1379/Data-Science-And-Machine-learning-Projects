def calc_up(beginning, destination, numbers, buttons, cost_list):
    minimum_sum = (max(cost_list) + 1) * 100
    if buttons[1] == 1:
        if True:
            pointer = beginning
            sum = cost_list[pointer]
            while True:
                pointer += 1
                pointer = pointer % len(cost_list)
                if pointer < 10 and numbers[pointer % 10] == 1:
                    sum = 1 + cost_list[pointer] + cost_list[beginning]
                else:
                    sum += cost_list[pointer] + 1
                if pointer == destination:
                    minimum_sum = min(sum, minimum_sum)
                    break
        if buttons[0] == 1:
            pointer = beginning
            sum = cost_list[pointer]
            while True:
                pointer += 1
                pointer = pointer % len(cost_list)
                if (pointer < 10 and numbers[pointer % 10] == 1) or (
                        pointer >= 10 and numbers[pointer % 10] == 1 and numbers[(pointer // 10) % 10] == 1):
                    if pointer < 10 and numbers[pointer % 10] == 1:
                        sum = 1 + cost_list[pointer] + cost_list[beginning]
                    elif pointer >= 10 and numbers[pointer % 10] == 1 and numbers[(pointer // 10) % 10] == 1:
                        sum = 3 + cost_list[pointer] + cost_list[beginning]
                else:
                    sum += cost_list[pointer] + 1
                if pointer == destination:
                    minimum_sum = min(sum, minimum_sum)
                    break
        else:
            pointer = beginning
            sum = cost_list[pointer]
            while True:
                pointer += 1
                pointer = pointer % len(cost_list)
                sum += cost_list[pointer] + 1
                if pointer == destination:
                    minimum_sum = min(sum, minimum_sum)
                    break
    if buttons[2] == 1:
        if True:
            pointer = beginning
            sum = cost_list[pointer]
            while True:
                pointer -= 1
                pointer = pointer % len(cost_list)
                if pointer < 10 and numbers[pointer % 10] == 1:
                    sum = 1 + cost_list[pointer] + cost_list[beginning]
                else:
                    sum += cost_list[pointer] + 1
                if pointer == destination:
                    minimum_sum = min(sum, minimum_sum)
                    break
        if buttons[0] == 1:
            pointer = beginning
            sum = cost_list[pointer]
            while True:
                pointer -= 1
                pointer = pointer % len(cost_list)
                if (pointer < 10 and numbers[pointer % 10] == 1) or (
                        pointer >= 10 and numbers[pointer % 10] == 1 and numbers[(pointer // 10) % 10] == 1):
                    if pointer < 10 and numbers[pointer % 10] == 1:
                        sum = 1 + cost_list[pointer] + cost_list[beginning]
                    elif pointer >= 10 and numbers[pointer % 10] == 1 and numbers[(pointer // 10) % 10] == 1:
                        sum = 3 + cost_list[pointer] + cost_list[beginning]
                else:
                    sum += cost_list[pointer] + 1
                if pointer == destination:
                    minimum_sum = min(sum, minimum_sum)
                    break
        else:
            pointer = beginning
            sum = cost_list[pointer]
            while True:
                pointer -= 1
                pointer = pointer % len(cost_list)
                sum += cost_list[pointer] + 1
                if pointer == destination:
                    minimum_sum = min(sum, minimum_sum)
                    break
    if buttons[0] == 0 and buttons[1] == 0 and buttons[2] == 0:
        minimum_sum = -1
    elif buttons[0] == 1 and numbers[destination % 10] == 1 and numbers[(destination // 10) % 10] == 1:
        minimum_sum == min(minimum_sum,
                           cost_list[beginning] + cost_list[destination] + (3 if destination >= 10 else 1))
    if buttons[0] == 1 and buttons[1] == 0 and buttons[2] == 0 and not (
            numbers[destination % 10] == 1 and numbers[(destination // 10) % 10] == 1):
        minimum_sum = -1
    if destination < 10 and numbers[destination] == 1:
        minimum_sum == min(minimum_sum,
                           cost_list[beginning] + cost_list[destination] + 1)
    print(minimum_sum)


numbers = list(map(int, input().split()))
buttons = list(map(int, input().split()))
cost_list = []
for i in range(10):
    cost_list.extend(list(map(int, input().split())))
beginning, destination = map(int, input().split())
calc_up(beginning, destination, numbers, buttons, cost_list)
