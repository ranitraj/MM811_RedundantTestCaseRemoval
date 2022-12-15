import constants


def get_unique_word_count(data):
    """
    Calculates the unique wordcount in the dataset

    :param data: input data
    :return: unique word count
    """
    words_list = list()
    test_steps = list(data[constants.STEPS])
    for cur_step in test_steps:
        for cur_word in cur_step:
            words_list.append(cur_word)
    return len(set(words_list))



