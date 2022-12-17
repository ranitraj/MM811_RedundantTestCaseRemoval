import pandas as pd

import constants
import preprocessing_utils
import model_word2vec


from gensim.models.keyedvectors import KeyedVectors

if __name__ == '__main__':
    # Part-1 [Performing Test-Step Similarity]

    # Load dataset
    training_data = pd.read_excel(constants.DATASET_PATH)

    # Preprocess the training dataset
    preprocessing_utils.read_input_data(training_data)
    test_steps_df = preprocessing_utils.clean_dataset(training_data)
    training_list = preprocessing_utils.return_training_list(test_steps_df)

    # Loading pre-trained on 15GB of Stack Overflow posts Word2Vec Model
    pre_trained_model_path = "SO_vectors_200.bin"
    pre_trained_model = KeyedVectors.load_word2vec_format(
        pre_trained_model_path,
        binary=True
    )

    # Train the word2vec model using the new dataset
    my_model = model_word2vec.train_word2vec_model(
        training_list,
        pre_trained_model,
        constants.VECTOR_DIMENSION,
        constants.WORKER_COUNT,
        constants.WORD_FREQUENCY_THRESHOLD,
        constants.CONTEXT_WINDOW_SIZE,
        constants.DOWN_SAMPLING
    )

