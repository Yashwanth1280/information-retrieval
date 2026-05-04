from __future__ import division
from sklearn.cluster import KMeans
from numbers import Number
from pandas import DataFrame
import sys, codecs, numpy
import os
import nltk
import re
import pyfpgrowth
import pandas as pd

class autovivify_list(dict):
    def __missing__(self, key):
        '''Given a missing key, set initial value to an empty list'''
        value = self[key] = []
        return value

    def __add__(self, x):
        '''Override addition for numeric types when self is empty'''
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        '''Also provide subtraction method'''
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError


def build_word_vector_matrix(vector_file, n_words):
    '''Return the vectors and labels for the first n_words in vector file'''
    numpy_arrays = []
    labels_array = []
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for c, r in enumerate(f):
            sr = r.split()
            labels_array.append(sr[0])
            numpy_arrays.append( numpy.array([float(i) for i in sr[1:]]) )
            if c == n_words:
                return numpy.array( numpy_arrays ), labels_array
    return numpy.array( numpy_arrays ), labels_array


def find_word_clusters(labels_array, cluster_labels):
    '''Return the set of words in each cluster'''
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[ i ].append( labels_array[c] )
    return cluster_to_words


# File and parameters
input_vector_file = r"C:/Users/jaide/OneDrive/Desktop/IR PROJECT/IR_Project/vectors.txt"
n_words = 1262  # Number of words to analyze 
reduction_factor = 0.1  # Amount of dimension reduction {0,1}
n_clusters = int(n_words * reduction_factor)  # Number of clusters to make

# Build word vector matrix
df, labels_array = build_word_vector_matrix(input_vector_file, n_words)

# KMeans clustering
kmeans_model = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans_model.fit(df)

cluster_labels = kmeans_model.labels_
cluster_inertia = kmeans_model.inertia_
cluster_to_words = find_word_clusters(labels_array, cluster_labels)

# Group clusters
clusters = []
for c in cluster_to_words:
    clusters.append(cluster_to_words[c])

# Prepare cluster dictionary
clust_dict = {}
for clus in clusters:
    if len(clus) == 1:
        if 0 not in clust_dict:
            clust_dict[0] = clus
        else:
            clust_dict[0] += clus
    else:
        clust_dict[clusters.index(clus)] = clus

# Print cluster information
for key, value in clust_dict.items():
    print(key, '    ', value)

import os
import nltk
import re

# Data preprocessing for mapping (Automated File Reading)
data_folder = r"C:/Users/jaide/OneDrive/Desktop/IR PROJECT/IR_Project/DATA-NLP"

# List all files in the folder
all_files = os.listdir(data_folder)

# Filter files with the expected pattern (e.g., ending with .txt)
files_to_read = [os.path.join(data_folder, file) for file in all_files if file.endswith('.txt')]

question_mapping = dict()  # Mapping to question dictionary

for path_to_file in files_to_read:
    split_name = os.path.basename(path_to_file).split('_')
    student_name = split_name[0]
    question_number = split_name[1].split('.')[0]
    
    if question_number not in question_mapping:
        question_mapping[question_number] = {}

    try:
        # Check if file exists before opening it
        if os.path.exists(path_to_file):
            with open(path_to_file, 'r', encoding='latin1') as f:
                mylist = f.read()
                sent_tokenize_list = nltk.sent_tokenize(mylist)
                temp = []
                for i in sent_tokenize_list:
                    sent = i.lower()
                    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
                    temp.append(sent)
                question_mapping[question_number][student_name] = temp
        else:
            print(f"File not found: {path_to_file}")
    except Exception as e:
        print(f"Error reading file {path_to_file}: {e}")

# Verify the structure of the question mapping
for question, student_data in question_mapping.items():
    print(f"Question {question}: {len(student_data)} students' answers processed.")




# Creating vector of sentences
vector_dict = {}
for key, value in question_mapping.items():
    vector_dict[key] = {}
    for student, answer in value.items():
        vector_dict[key][student] = []
        for sent in answer:
            words = sent.split(' ')
            temp_list = []
            for w in words:
                for cluster_key, cluster_value in clust_dict.items():
                    if w in clust_dict[cluster_key]:
                        temp_list.append(cluster_key)
            vector_dict[key][student].append(temp_list)

for k, v in vector_dict.items():
    print(k, '    ', v)
    break


# Create vector name mapping
vector_name = {}
rep = 0
for key, value in vector_dict.items():
    for student, vector in value.items():
        for v in vector:
            vector_name[tuple(v)] = rep
            rep += 1

# Creating answer vectors
pro_dict = {}
for key, value in vector_dict.items():
    pro_dict[key] = {}
    for student, answer in value.items():
        pro_dict[key][student] = []
        for v in answer:
            ans = tuple(v)
            pro_dict[key][student].append(vector_name[ans])

for key, value in pro_dict.items():
    print(key, '    ', value)
    break

print(len(pro_dict.keys()))


# FP-Growth Algorithm
inp = ["taske", "taska", "taskb", "taskc", "taskd"]

def fp_growth(transactions):
    patterns = pyfpgrowth.find_frequent_patterns(transactions, 3)
    frequent_list = []
    for p in patterns:
        if len(p) < 3:
            continue
        else:
            frequent_list.append(list(p))
    return frequent_list


def is_sub(sub, lst):
    ln = len(sub)
    for i in range(len(lst) - ln + 1):
        if all(sub[j] == lst[i+j] for j in range(ln)):
            return True
    return False

def output(frequent_list):
    output_dict= {}
    for lst in frequent_list:
        for k, v in pro_dict.items():
            for name, vec in v.items():
                if is_sub(lst,vec):
                    if k not in output_dict:
                        output_dict[k] = [name]
                    else:
                        output_dict[k].append(name)
    return output_dict


# Final output from our script
final= {}
for task in inp:
    transactions = []
    for k, v in pro_dict.items():
        if k == task:
            for name, vec in v.items():
                transactions.append(vec)
            frequent_list = fp_growth(transactions)
    output_dict = output(frequent_list)
    for key, value in output_dict.items():
        final[key] = set(value)


for i, v in final.items():
    print(i, '    ', v)


# Evaluation
xl = pd.ExcelFile(r"C:/Users/jaide/OneDrive/Desktop/IR PROJECT/IR_Project/corpus_final.xls")
df = xl.parse("File list")
c = df[['File', 'Category']]

tp = 0
tn = 0
fn = 0
fp = 0
for key, value in final.items():
    for v in value:
        ans = v + key + '.txt'
        head = key
        for index, row in c.iterrows():
            if row['File'].split('_')[1].split('.')[0] == head:
                if row['File'].split('_')[0] == v:
                    if row['Category'] == 'cut' or row['Category'] == 'heavy':
                        tp += 1
                    else:
                        fp += 1
                else:
                    if row['Category'] == 'cut' or row['Category'] == 'heavy':
                        fn += 1
                    else:
                        tn += 1

precision = float(tp / (tp + fp))                        
print('Precision is : ', precision*100, '%')
recall = float(tp / (tp + fn))
print('Accuracy is : ', ((tp + tn) / (tp + fn + tn + fp))*100 , '%')
print('recall is : ', recall*100 , '%')
print('F1-score is : ', ((2* precision * recall) / (precision + recall))*100, '%')