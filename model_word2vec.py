import numpy as np
import constants
import gensim

from gensim.similarities import WmdSimilarity
from gensim.models import Word2Vec, Phrases, KeyedVectors


def train_word2vec_model(data, pre_trained_model, vector_size, workers, word_freq_threshold,
                         context_window_size, down_sampling):
    """
    Trains and adds the new corpus into the pretrained word2vec model

    :param data: preprocessed data
    :param pre_trained_model:
    :param vector_size: vector size dimension
    :param workers: number of workers
    :param word_freq_threshold: maximum frequency limit for a word
    :param context_window_size: context size of CBOW
    :param down_sampling: down-sampling value
    :return: trained model
    """
    list_word_vector_median = list()

    # Initialize model
    my_model = Word2Vec(
        vector_size=vector_size,
        workers=workers,
        min_count=word_freq_threshold,
        window=context_window_size,
        sample=down_sampling
    )

    # Build model and count total number of examples
    my_model.build_vocab(data)
    total_examples = my_model.corpus_count
    all_words = list(my_model.wv.index_to_key)
    for cur_word in all_words:
        my_model.wv[cur_word] = np.zeros(constants.VECTOR_DIMENSION)

    # Update vocabulary with our corpus
    my_model.build_vocab([list(pre_trained_model.index_to_key)], update=True)
    my_model.wv.vectors_lockf = np.ones(len(my_model.wv))
    my_model.wv.intersect_word2vec_format(pre_trained_model, binary=True, lockf=1.0)

    # Get Mean & Standard-Deviation of initialized word vectors
    for cur_word in all_words:
        if any(my_model.wv[cur_word] != 0):
            list_word_vector_median.append(np.median(my_model.wv[cur_word]))

    # Initialize Mean & Standard-Deviation for normal distributions of medians
    mu = np.mean(list_word_vector_median)
    sigma = np.std(list_word_vector_median)

    # Initialize the remaining word vectors that are not present in the pre-trained word2vec model
    for cur_word in all_words:
        if all(my_model.wv[cur_word] == 0):
            new_word_vector = np.random.normal(mu, sigma, 200)
            my_model.wv[cur_word] = new_word_vector

    # Train the model
    my_model.train(data, total_examples=total_examples, epochs=25)
    return my_model
