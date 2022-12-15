def get_unique_word_count(data, column_name):
    """
    Calculates the unique wordcount in the dataset from a particular column

    :param data: input data
    :param column_name: name of the column from which unique word count is calculated
    :return: unique word count
    """
    list_words = list()
    column = list(data[column_name])
    for cur_step in column:
        for cur_word in cur_step:
            list_words.append(cur_word)
    return len(set(list_words))


def get_word_frequency(data, column_name):
    """
    Returns the list of words that have a frequency less than two in the dataset

    :param data: input data
    :param column_name: name of the column from which word frequency is calculated
    :return: list of words
    """
    list_words = list()
    column = list(data[column_name])
    for cur_step in column:
        for cur_word in cur_step:
            list_words.append(cur_word)
    unique_words_list = set(list_words)

    dict_word_frequency = {}
    for cur_word in unique_words_list:
        # Initialize count of all words to 0
        dict_word_frequency[cur_word] = 0

    for cur_step in column:
        # Compute frequency of each word
        for cur_word in cur_step:
            dict_word_frequency[cur_word] += 1

    final_list = list()
    # List of words that occur only once
    for word, frequency in dict_word_frequency.items():
        if frequency < 2:
            final_list.append(word)

    return final_list
