import re
import string
import nltk
import constants

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


def clean_dataset(data):
    """
    Cleans the input dataset

    :param data: input data
    :return: cleaned dataset
    """
    print("Size of dataset before preprocessing = ", data.shape)

    # Replace URL, paths, HTML tags and convert to lower-case
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: re.sub(r'http\S+', 'URL', x))
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: re.sub(r'/[\w-]*', '', x))
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: re.sub(r'\{[^)]*}', '', x))
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: x.lower())

    # Remove digits, punctuations and extra-spaces
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: re.sub('\w*\d\w*', '', x))
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))
    data[constants.STEPS] = data[constants.STEPS].apply(lambda x: re.sub(' +', ' ', x))

    # Tokenization for test case-names
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(lambda x: TweetTokenizer().tokenize(x))
    unique_word_count = get_unique_word_count(data, constants.CASE_NAME)
    print("Number of unique words across all test names: ", unique_word_count)

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(lambda x: [w for w in x if not w in stop_words])
    unique_word_count = get_unique_word_count(data, constants.CASE_NAME)
    print("Number of unique words in test names after stopword removal: ", unique_word_count)

    # Lemmatization for test case-names
    word_net_lemmatizer = WordNetLemmatizer()
    data[constants.CASE_NAME] = data[constants.CASE_NAME].apply(lambda x: [word_net_lemmatizer.lemmatize(w) for w in x])

    # Remove words that occur a certain number of times
    word_frequency_threshold = get_word_frequency(data, constants.CASE_NAME)
    print("Number of words that occurred only once in test case names: ", len(word_frequency_threshold))

    # list of words to be removed
    for index, row in data.iterrows():
        current_test_name = row[constants.CASE_NAME]
        list_words_to_remove = list()
        for word in current_test_name:
            if word in word_frequency_threshold:
                list_words_to_remove.append(word)
        data.loc[index][constants.CASE_NAME] = [elem for elem in current_test_name if not elem in list_words_to_remove]

    # Remove instances with empty names
    data = data.loc[data[constants.CASE_NAME] != '']

    unique_word_count = get_unique_word_count(data, constants.CASE_NAME)
    print("Unique word count  = ", unique_word_count)
    print("Size of dataset after preprocessing = ", data.shape)
