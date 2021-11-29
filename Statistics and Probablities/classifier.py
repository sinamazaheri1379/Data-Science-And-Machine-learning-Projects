import re
from shutil import copyfile

number_of_ham_emails = 300
number_of_spam_emails = 300
total_emails = number_of_ham_emails + number_of_spam_emails


def initialize_the_table(size):    # size is the default length of the email
    the_table = list()
    for i in range(size):
        the_table.append((dict(), dict()))
    return the_table


def get_the_word_probability_table_by_emails(path_address, class_num, table, number_of_email):
    for i in range(1, number_of_email + 1):
        email = open(path_address + '(' + str(i) + ')' + ".txt", 'r', encoding='utf-8-sig')
        email_content = email.read()
        list_of_words = list(filter(lambda x: len(x) != 0, re.split("\\s+", email_content)))  # filter the white
        # characters
        iteration_number = min(len(table), len(list_of_words))  # choose the minimum length between default and n
        for j in range(iteration_number):  # compute probabilities
            feature_dic = table[j][class_num]
            word = list_of_words[j]
            feature_dic[word] = feature_dic.get(word, 0) + (1 / number_of_email)
        for j in range(len(table) - iteration_number):  # fill the rest of the length with blank
            feature_dic = table[j + iteration_number][class_num]
            feature_dic[''] = feature_dic.get('', 0) + (1 / number_of_email)
        email.close()
    return table


def classify_the_emails(source_path, number_of_emails, table, total_training_emails, number_of_hamtraining_emails,
                        number_of_spamtraining_emails):
    if "ham" in source_path:
        class_name = "ham"
    else:
        class_name = "spam"
    n = len(table)
    for i in range(1, number_of_emails + 1):
        target_path = '(' + str(i) + ')' + ".txt"
        target_email = open(source_path + target_path, 'r', encoding="utf-8-sig")
        target_content = list(filter(lambda x: len(x) != 0, re.split("\\s+", target_email.read())))
        iteration_number = min(n, len(target_content))
        probability_of_being_E_respect_to_condition_H = 1  # P(E|H)
        probability_of_being_E_respect_to_condition_S = 1  # P(E|S)
        for j in range(iteration_number):
            word = target_content[j]
            feature_dic_ham = table[j][0]
            feature_dic_spam = table[j][1]
            p_0 = feature_dic_ham.get(word, 0)
            p_1 = feature_dic_spam.get(word, 0)
            if p_0 == 0 or p_1 == 0:
                continue
            probability_of_being_E_respect_to_condition_H = p_0 * probability_of_being_E_respect_to_condition_H
            probability_of_being_E_respect_to_condition_S = p_1 * probability_of_being_E_respect_to_condition_S
        target_email.close()
        evidence_probability = (
                                           probability_of_being_E_respect_to_condition_H * number_of_hamtraining_emails / total_training_emails) + \
                               (
                                           probability_of_being_E_respect_to_condition_S * number_of_spamtraining_emails / total_training_emails)
        # compute evidence probability = P(E) = P(E|H)P(H) + P(E|S)P(S)
        probability_of_being_H_respect_to_condition_E = ((number_of_hamtraining_emails / total_training_emails) *
                                                         probability_of_being_E_respect_to_condition_H) / evidence_probability
        # compute P(H|E) = P(E|H)P(H) / P(E)
        probability_of_being_S_respect_to_condition_E = 1 - probability_of_being_H_respect_to_condition_E  # compute P(S|E) = 1 - P(H|E)
        if probability_of_being_H_respect_to_condition_E > probability_of_being_S_respect_to_condition_E:
            copyfile(source_path + target_path, ".\\emails\\ham prediction\\" + class_name +
                     target_path)
        else:
            copyfile(source_path + target_path, ".\\emails\\spam prediction\\" + class_name
                     + target_path)


path = ".\\emails"
the_table_instance = initialize_the_table(40)
the_table_instance = get_the_word_probability_table_by_emails(path + "\\hamtraining\\hamtraining ", 0,
                                                              the_table_instance, number_of_ham_emails)
the_table_instance = get_the_word_probability_table_by_emails(path + "\\spamtraining\\spamtraining ", 1,
                                                              the_table_instance, number_of_spam_emails)
classify_the_emails(path + "\\hamtesting\\hamtesting ", 200, the_table_instance, 600, 300, 300)
classify_the_emails(path + "\\spamtesting\\spamtesting ", 200, the_table_instance, 600, 300, 300)

