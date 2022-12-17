import pandas as pd
import constants
import preprocessing_utils

if __name__ == '__main__':
    # Part-1 [Performing Test-Step Similarity]

    # Load dataset
    training_data = pd.read_excel(constants.DATASET_PATH)

    # Preprocess the training dataset
    preprocessing_utils.read_input_data(training_data)
    test_steps_df = preprocessing_utils.clean_dataset(training_data)
    training_list = preprocessing_utils.return_training_list(test_steps_df)


