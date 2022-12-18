import numpy as np
import constants

from sklearn.cluster import KMeans
from gensim.models import Word2Vec


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


def return_data_tuple(data):
    """
    Returns tuples with step_id, step which is used to retrieve the step_id after clustering

    :param data: data
    :return: list_tuple_step_id, list_test_step_clustering
    """
    list_tuple_step_id = list()
    list_test_step_clustering = list()

    for index, row in data.iterrows():
        step_id = row[constants.STEP_ID]
        step = row[constants.STEPS]
        list_tuple_step_id.append((step_id, step))

        temp_list = list()
        if isinstance(row[constants.STEPS], list):
            for cur_element in row[constants.STEPS]:
                temp_list.append(cur_element)
        else:
            if isinstance(row[constants.STEPS], str):
                temp_list.append(row[constants.STEPS])
        list_test_step_clustering.append(temp_list)

    print(f"First Tuple = {list_tuple_step_id[0]}")
    return list_tuple_step_id, list_test_step_clustering


def initialize_similarity_matrix(list_test_step_clustering):
    """
    Create and returns an empty matrix with rows and columns equal to the shape of 'list_test_step_clustering'

    :param list_test_step_clustering: clustering steps list
    :return: matrix_similarity_distance
    """
    rows = columns = len(list_test_step_clustering)
    return np.zeros((rows, columns))


def compute_and_save_similarity_distance(model, list_test_step_clustering, matrix_similarity_distance):
    """
    Computes the similarity distance between the rows and columns of list_test_step_clustering
    and saves the result in a .txt file

    :param model: word2vec model
    :param list_test_step_clustering: clustering steps list
    :param matrix_similarity_distance: empty matrix
    :return: matrix_similarity_distance with similarities
    """
    rows = columns = len(list_test_step_clustering)

    for cur_row in range(rows):
        for cur_column in range(cur_row, columns):
            similarity_distance = model.wv.wmdistance(list_test_step_clustering[cur_row],
                                                      list_test_step_clustering[cur_column])
            # Upper limit
            if similarity_distance > 800:
                similarity_distance = 800
            matrix_similarity_distance[cur_row, cur_column] = matrix_similarity_distance[
                cur_column, cur_row] = similarity_distance

    # Save the similarity matrix
    path_save_matrix_similarity = 'matrix_similarity.txt'
    np.savetxt(path_save_matrix_similarity, matrix_similarity_distance)

    return matrix_similarity_distance


def compute_average_test_step(test_step, model, pre_trained_model):
    """
    Computes the average word vector of the test step

    :param test_step: current test-step
    :param model: trained word2vec model
    :param pre_trained_model: pre-trained word2vec model
    :return: average
    """
    list_word_vectors = list()
    count_word = 0
    for word in test_step:
        try:
            list_word_vectors.append(model[word])
        except:
            try:
                list_word_vectors.append(pre_trained_model[word])
            except:
                return np.zeros(200)
        count_word += 1
    sum_vectors = sum(list_word_vectors)
    average = sum_vectors / count_word
    return average


def perform_clustering_and_save(model, pre_trained_model, data, list_tuple_step_id, list_test_step_clustering):
    """
    Performs k-means clustering on the test_step and step_id and saves the clustered results in .txt files

    :param model: trained word2vec model
    :param pre_trained_model: pre-trained word2vec model
    :param data: processed data
    :param list_tuple_step_id: tuple of step-id
    :param list_test_step_clustering: clustered list
    :return: dict_clusters: dictionary of the clusters
    """
    labels_list_kmeans = list()

    # Compute average word vector to be used in k-means
    avg_word_sentence_vectors = list()
    for cur_test_step in list_test_step_clustering:
        if len(cur_test_step) > 0:
            avg_word_sentence_vectors.append(
                compute_average_test_step(
                    cur_test_step,
                    model,
                    pre_trained_model
                )
            )

    # Initialize K-means
    k_means = KMeans(
        n_clusters=constants.K_MEANS_CLUSTER_COUNT,
        init='k-means++',
        max_iter=500
    )
    k_means.fit(avg_word_sentence_vectors)
    labels = k_means.labels_
    labels_list_kmeans.append(labels)

    dict_clusters = {}
    for cur_label in set(labels):
        label_indices = np.where(labels == cur_label)[0].tolist()
        for cur_index in label_indices:
            dict_clusters[int(list_tuple_step_id[cur_index][0])] = cur_label

    print(f"Number of clusters = {k_means.n_clusters} & Number of labels = {k_means.labels_}")
    save_clusters(k_means.labels_, list_tuple_step_id, list_test_step_clustering, data)
    save_cluster_labels(k_means.labels_, list_tuple_step_id)

    return dict_clusters


def save_clusters(labels, list_tuple_step_id, list_test_step_clustering, data):
    """
    Saves clusters in a text-file

    :param labels: k-means labels
    :param list_tuple_step_id: tuple of step-id
    :param list_test_step_clustering: clustered list
    :param data: processed data
    """
    path = "clustered_data_kmeans.txt"
    output = open(path, "a")
    for label in set(labels):
        indices_label = np.where(labels == label)[0].tolist()
        for index in indices_label:
            str_to_save = "[" + str(label) + "]:\t\t" + data.loc[index]["Key"] + "\t\t" + str(
                list_tuple_step_id[index][0]) + "\t\t" + str(list_test_step_clustering[index]) + "\n"
            output.write(str_to_save)
    output.close()


def save_cluster_labels(labels, list_tuple_step_id):
    """
    Saves cluster labels in a text-file

    :param labels: k-means labels
    :param list_tuple_step_id: tuple of step-id
    """
    path = "clustered_labels_kmeans.txt"
    output = open(path, "a")
    for single_label in set(labels):
        indices_label = np.where(labels == single_label)[0].tolist()
        str_to_save = "[" + str(single_label) + "]: " + ','.join(
            str(list_tuple_step_id[x][0]) for x in indices_label) + "\n"
        output.write(str_to_save)
    output.close()
