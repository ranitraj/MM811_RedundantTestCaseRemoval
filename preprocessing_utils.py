import re
import string
import nltk
import constants
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


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


def read_input_data(data):
    """
    Method to load input data and iterate through it

    :param data: training data
    """
    print("Reading input data")
    cur_index = 0
    for index, row in data.iterrows():
        current_type = row[constants.TYPE]
        current_key = row[constants.KEY]
        current_name = row[constants.CASE_NAME]
        current_step_id = row[constants.STEP_ID]
        current_steps = row[constants.STEPS]

        data.loc[cur_index] = [current_type, current_key, current_name, current_step_id, current_steps]
        cur_index += 1
    print("Shape of data = ", data.shape)


def clean_dataset(data):
    """
    Cleans the input dataset for the test-steps and test-case columns

    :param data: input data
    :return: cleaned dataset
    """
    print("Size of dataset before preprocessing = ", data.shape)

    # Replace URL, paths, HTML tags and convert to lower-case
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: re.sub(r'http\S+', 'URL', x))
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(lambda x: re.sub(r'http\S+', 'URL', x))
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: re.sub(r'/[\w-]*', '', x))
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(lambda x: re.sub(r'/[\w-]*', '', x))
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: re.sub(r'\{[^)]*}', '', x))
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(lambda x: re.sub(r'\{[^)]*}', '', x))
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: x.lower())
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(lambda x: x.lower())

    # Remove digits, punctuations and extra-spaces
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: re.sub('\w*\d\w*', '', x))
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(lambda x: re.sub('\w*\d\w*', '', x))
    data[constants.STEPS] = data[constants.STEPS].apply(
        lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(
        lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: re.sub(' +', ' ', x))
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(lambda x: re.sub(' +', ' ', x))

    # Tokenization
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: TweetTokenizer().tokenize(x))
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(lambda x: TweetTokenizer().tokenize(x))
    unique_word_count_steps = get_unique_word_count(data, constants.STEPS)
    unique_word_count_case = get_unique_word_count(data, constants.CASE_NAME)
    print(f"Number of unique words across Test-cases = {unique_word_count_case} "
          f"& Test-Steps = {unique_word_count_steps} after Tokenization")

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: [w for w in x if not w in stop_words])
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(lambda x: [w for w in x if not w in stop_words])
    unique_word_count_steps = get_unique_word_count(data, constants.STEPS)
    unique_word_count_case = get_unique_word_count(data, constants.CASE_NAME)
    print(f"Number of unique words across Test-cases = {unique_word_count_case} "
          f"& Test-Steps = {unique_word_count_steps} after Stopword Removal")

    # Lemmatization for test case-names
    word_net_lemmatizer = WordNetLemmatizer()
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: [word_net_lemmatizer.lemmatize(w) for w in x])
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(lambda x: [word_net_lemmatizer.lemmatize(w) for w in x])

    # Remove words that occur a certain number of times
    word_frequency_threshold_steps = get_word_frequency(data, constants.STEPS)
    word_frequency_threshold_case = get_word_frequency(data, constants.CASE_NAME)
    print(f"Number of words that occurred only once in Test-Cases = {len(word_frequency_threshold_case)}"
          f"& Test-Steps = {len(word_frequency_threshold_steps)}")

    # List of words to be removed in Test-Steps
    for index, row in data.iterrows():
        current_test_name = row[constants.STEPS]
        list_words_to_remove_steps = list()
        for word in current_test_name:
            if word in word_frequency_threshold_case:
                list_words_to_remove_steps.append(word)
        data.loc[index][constants.STEPS] = [elem for elem in current_test_name if
                                            not elem in list_words_to_remove_steps]

    # List of words to be removed in Test-Case
    for index, row in data.iterrows():
        current_test_name = row[constants.CASE_NAME]
        list_words_to_remove_case = list()
        for word in current_test_name:
            if word in word_frequency_threshold_case:
                list_words_to_remove_case.append(word)
        data.loc[index][constants.CASE_NAME] = [elem for elem in current_test_name if
                                                not elem in list_words_to_remove_case]

    # Remove instances with empty names
    data = data.loc[data[constants.STEPS] != '']
    data = data.loc[data[constants.CASE_NAME] != '']

    unique_word_count_steps = get_unique_word_count(data, constants.STEPS)
    unique_word_count_case = get_unique_word_count(data, constants.CASE_NAME)
    print(f"Unique word count in Test-Case = {unique_word_count_case} "
          f"& Unique word count in Test-Steps = {unique_word_count_steps}")

    print("Size of dataset after preprocessing = ", data.shape)
    return data


def return_training_list(data):
    """
    Returns the necessary fields to train the word2vec word embedding model i.e 'type', 'name', 'steps'

    :param data: preprocessed data
    :return: training data
    """
    training_list = list()
    for index, row in data.iterrows():
        temp_list = list()

        if not pd.isnull(row[constants.TYPE]):
            temp_list.append(str(row[constants.TYPE]))

        if isinstance(row[constants.CASE_NAME], list):
            for elem in row[constants.CASE_NAME]:
                temp_list.append(elem)
        else:
            if isinstance(row[constants.CASE_NAME], str):
                temp_list.append(row[constants.CASE_NAME])

        if isinstance(row[constants.STEPS], list):
            for elem in row[constants.STEPS]:
                temp_list.append(elem)
        else:
            if isinstance(row[constants.STEPS], str):
                temp_list.append(row[constants.STEPS])

        # List of lists of tokens
        training_list.append(temp_list)
    print(f"Length of list with training data = {len(training_list)}")
    return training_list
