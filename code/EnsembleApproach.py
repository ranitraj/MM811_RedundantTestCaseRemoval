
import os
import gc
import pandas as pd
import numpy as np
import math
import statistics as st
import re
import string
import time
import matplotlib.pyplot as plt
from collections import defaultdict  # For word frequency

from nltk.corpus import stopwords |
from nltk.tokenize import RegexpTokenizer, word_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer
import nltk



appr_clusters_dict_1 = {}
cluster_file = open('/content/appr_1_kmeans_cluster_labels.txt')
for line in cluster_file:
    full_line = line.split()
    cluster_id = int(full_line[0].replace('[', '').replace(']', '').replace(':', ''))
    step_id_list = full_line[1].split(',')
    for step_id in step_id_list:
        appr_clusters_dict_1[int(float(step_id))] = cluster_id

print("Number of test steps which were clustered by the approach: ", len(appr_clusters_dict_1))

appr_clusters_dict_2 = {}
cluster_file = open('/content/appr_1_cluster_labels.txt')
for line in cluster_file:
    full_line = line.split()
    cluster_id = int(full_line[0].replace('[', '').replace(']', '').replace(':', ''))
    step_id_list = full_line[1].split(',')
    for step_id in step_id_list:
        appr_clusters_dict_2[int(float(step_id))] = cluster_id

print("Number of test steps which were clustered by the approach: ", len(appr_clusters_dict_1))


def get_number_unique_words(df):
    words_list = list()
    test_steps = list(df["Steps"])
    for step in test_steps:
        for word in step:
            words_list.append(word)
    number_unique_words = len(set(words_list))
    return number_unique_words


def get_word_frequency(df):
    words_list = list()
    test_steps = list(df["Steps"])
    for step in test_steps:
        for word in step:
            words_list.append(word)
    unique_words_list = set(words_list)
    word_occurrence_dict = {}
    for each_word in unique_words_list:
        word_occurrence_dict[each_word] = 0

    for step in test_steps:
        for word in step:
            word_occurrence_dict[word] += 1

    ten_times_occurrence_words = list()
    # get list of words that occur only once
    for word, occurrence in word_occurrence_dict.items():
        if occurrence < 2:
            ten_times_occurrence_words.append(word)

    return ten_times_occurrence_words


def remove_problematic_words(df):
    number_unique_words = get_number_unique_words(df)
    print("Number of unique words across all test steps: ", number_unique_words)

    problematic_words = open('word2vec_vocab_problematic.txt', 'r')
    problematic_words_list = list()
    for word in problematic_words:
        problematic_words_list.append(word.lstrip().rstrip())

    for index, row in df.iterrows():
        step = row["Steps"]
        df.loc[index]["Steps"] = [elem for elem in step if not elem in problematic_words_list]

    number_unique_words = get_number_unique_words(df)
    print("Number of unique words across all test steps after removing problematic words: ", number_unique_words)


def fix_problematic_words(df):
    number_unique_words = get_number_unique_words(df)
    print("Number of unique words across all test steps: ", number_unique_words)


    problematic_words = open('word2vec_vocab_to_fix.txt', 'r')
    problematic_words_dict = {}
    for line in problematic_words:
        full_line = line.split(':')
        try:
            problematic_words_dict[full_line[0]] = [x.replace('\n', '') for x in full_line[1].split(',')]
        except:
            problematic_words_dict[full_line[0]] = full_line[1].replace('\n', '')

    for index, row in df.iterrows():
        step = row["Steps"]
        modified_step = list()
        for word in step:
            if word in problematic_words_dict:
                modified_step.extend(problematic_words_dict[word])
            else:
                modified_step.append(word)
        df.loc[index]["Steps"] = modified_step

    number_unique_words = get_number_unique_words(df)
    print("Number of unique words across all test steps after fixing problematic words: ", number_unique_words)


def preprocess_clean_data(df):
    print("Cleaning test step field...")

    df["Steps"] = df["Steps"].apply(lambda x: re.sub(r'http\S+', 'URL', x))
    df["Steps"] = df["Steps"].apply(lambda x: re.sub('\/[\w-]*', '', x))
    df["Steps"] = df["Steps"].apply(lambda x: re.sub(r'\{[^)]*\}', '', x))


    df["Steps"] = df["Steps"].apply(lambda x: x.lower())


    df["Steps"] = df["Steps"].apply(lambda x: re.sub('\w*\d\w*', '', x))


    df["Steps"] = df["Steps"].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x))


    df["Steps"] = df["Steps"].apply(lambda x: re.sub(' +', ' ', x))


    df["Steps"] = df["Steps"].apply(lambda x: TweetTokenizer().tokenize(x))
    number_unique_words = get_number_unique_words(df)
    print("Number of unique words across all test steps: ", number_unique_words)


    ten_times_occurrence_words = get_word_frequency(df)
    print("Number of words that occurred less than 10 times in test steps: ", len(ten_times_occurrence_words))

    for index, row in df.iterrows():
        current_test_step = row["Steps"]
        list_words_to_remove = list()
        for word in current_test_step:
            if word in ten_times_occurrence_words:
                list_words_to_remove.append(word)

        test_steps_df.loc[index]["Steps"] = [elem for elem in current_test_step if not elem in list_words_to_remove]

    print("Dataset size after preprocessing: ", df.shape)


current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir) + "\\filtered_data\\"
xlsxfiles = [os.path.join(root, name)
             for root, dirs, files in os.walk(parent_dir)
             for name in files
             if name.endswith((".xlsx"))]

column_names = ["Type", "Key", "Case_Name", "Step_ID", "Steps"]
test_steps_df = pd.DataFrame(columns=column_names)

index_to_add = 0

print("Reading input data...")
test_file = '/content/drive/MyDrive/test_cases.xlsx'
test_data_df = pd.read_excel(test_file)
for index, row in test_data_df.iterrows():
    current_type = row["Type"]
    current_key = row["Key"]
    current_name = row["Case_Name"]
    current_step_id = row["Step_ID"]
    current_steps = row["Steps"]
    test_steps_df.loc[index_to_add] = [current_type, current_key, current_name, current_step_id, current_steps]
    index_to_add += 1

print("Done!")
print("Shape of data => ", test_steps_df.shape)

import nltk

nltk.download('stopwords')

preprocess_clean_data(test_steps_df)


step_id_text_tuple_list = list()
test_steps_clustering_list = list()
for index, row in test_steps_df.iterrows():
    step_id = row["Step_ID"]
    step_text = row["Steps"]
    step_id_text_tuple_list.append((step_id, step_text))

    temp_list = list()
    if isinstance(row["Steps"], list):
        for elem in row["Steps"]:
            temp_list.append(elem)
    else:
        if isinstance(row["Steps"], str):
            temp_list.append(row["Steps"])

    test_steps_clustering_list.append(temp_list)

print("Length of list of tuples:", len(step_id_text_tuple_list))
print("Length of list with test steps: ", len(test_steps_clustering_list))

# Remove empty steps
index = 0
steps_to_remove = list()
for step in test_steps_clustering_list:
    if len(step) == 0:
        steps_to_remove.append(index)
    index += 1

step_id_text_tuple_list = [step_id_text_tuple_list[index] for index in range(len(step_id_text_tuple_list)) if
                           not index in steps_to_remove]
test_steps_clustering_list = [test_steps_clustering_list[index] for index in range(len(test_steps_clustering_list)) if
                              not index in steps_to_remove]
print("Length of list of tuples:", len(step_id_text_tuple_list))
print("Length of list with test steps: ", len(test_steps_clustering_list))


clusters_list = []
found_flag = [False] * len(test_steps_clustering_list)

for i in range(len(test_steps_clustering_list) - 1):
    temp_set = set()
    if not found_flag[i]:
        temp_set.add(i)
        found_flag[i] = True
    else:
        continue

    for j in range(i + 1, len(test_steps_clustering_list)):
        if found_flag[j]:
            continue
        else:

            step_id_1 = step_id_text_tuple_list[i][0]
            step_id_2 = step_id_text_tuple_list[j][0]
            if ((appr_clusters_dict_1[step_id_1] == appr_clusters_dict_1[step_id_2]) + (
                    appr_clusters_dict_2[step_id_1] == appr_clusters_dict_2[step_id_2])) >= 3:
                temp_set.add(j)
                found_flag[j] = True
    clusters_list.append(temp_set)


print(len(clusters_list))

# Save
path_save_data = "/content/ensemble_clustered_data.txt"
out_cluster_file = open(path_save_data, "a")
cluster_id = 0

for cluster in clusters_list:
    for index in cluster:
        str_to_save = "[" + str(cluster_id) + "]:\t\t" + test_steps_df.loc[index]["Key"] + "\t\t" + str(
            step_id_text_tuple_list[index][0]) + "\t\t" + str(test_steps_clustering_list[index]) + "\n"
        out_cluster_file.write(str_to_save)
    cluster_id += 1
out_cluster_file.close()


path_save_labels = "/content/ensemble_cluster_labels.txt"
out_cluster_file = open(path_save_labels, "a")
cluster_id = 0
for cluster in clusters_list:
    str_to_save = "[" + str(cluster_id) + "]: " + ','.join(
        str(step_id_text_tuple_list[x][0]) for x in list(cluster)) + "\n"
    out_cluster_file.write(str_to_save)
    cluster_id += 1
out_cluster_file.close()

appr_ensemble_clusters_dict = {}
cluster_id = 0
for each_set in clusters_list:
    for index in each_set:
        step_id = step_id_text_tuple_list[index][0]
        appr_ensemble_clusters_dict[int(step_id)] = cluster_id
    cluster_id += 1

print("Number of test steps which were clustered by the approach: ", len(appr_ensemble_clusters_dict))

