import pickle
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import random
from itertools import product
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, make_scorer
from sklearn.model_selection import KFold, cross_validate
import xgboost as xgb

random.seed(0)

# features
classes = [0, 1, 2, 3]

prot_mapping = {'R': 0, 'K': 0, 'H': 0,
                'N': 1, 'W': 1, 'S': 1, 'Q': 1, 'Y': 1, 'G': 1, 'T': 1,
                'P': 2, 'M': 2, 'F': 2, 'D': 2, 'A': 2, 'V': 2, 'L': 2, 'I': 2,
                'C': 3, 'E': 3}

rna_mapping = {'A': 0, 'a': 0,
               'U': 1, 'u': 1, 'T': 1, 't': 1,
               'G': 2, 'g': 2,
               'C': 3, 'c': 3}

k = 5


def all_repeat(arr, k):
    results = []
    for c in product(arr, repeat=k):
        results.append(c)
    return results


prot_data_features = all_repeat(classes, k)
rna_data_features = all_repeat(classes, k)


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def flip_dict(original_dict):
    flipped = {}
    for key, value in original_dict.items():
        if value not in flipped:
            flipped[value] = [key]
        else:
            flipped[value].append(key)
    return flipped


def count_kmers(read, k):
    counts = {}
    num_kmers = len(read) - k + 1
    for i in range(num_kmers):
        kmer = read[i:i + k]
        if kmer not in counts:
            counts[kmer] = 0
        counts[kmer] += 1
    return counts


def feature_extract(kmers, mapping_type, sample_type_features):
    msu_rep = []
    sample_feature_dict = dict.fromkeys(sample_type_features, 0)

    for i in range(0, len(kmers)):
        curr_mer = kmers[i]
        temp = []
        for i in range(0, len(curr_mer)):
            temp.append(mapping_type[curr_mer[i]])
        msu_rep.append(tuple(temp))

    for i in range(0, len(msu_rep)):
        if msu_rep[i] in sample_feature_dict.keys():
            sample_feature_dict[msu_rep[i]] += 1

    return sample_feature_dict, msu_rep


def read_train_data(file_name):
    r_rep = pd.read_csv(file_name)
    r_rep = r_rep[['protein_seq', 'rna_seq', 'y_val']]

    prot = list(r_rep['protein_seq'].values)
    rna = list(r_rep['rna_seq'].values)
    y_train = list(r_rep['y_val'].values)
    data = []
    for i in range(0, len(y_train)):
        data.append([prot[i], rna[i], y_train[i]])
    random.shuffle(data)
    prot_list, rna_list, y_train_list = [], [], []
    for i in range(0, len(y_train)):
        prot_list.append(data[i][0])
        rna_list.append(data[i][1])
        y_train_list.append(data[i][2])
    print("Length of data  =", len(rna))

    return prot_list, rna_list, y_train_list


def train_the_model_and_calculate_metrics():
    # Prepare TRAIN DATA

    train_file = './R_Combine_70% (1).csv'
    train_prot, train_rna, y_train = read_train_data(file_name=train_file)

    # Prepare TEST DATA

    test_file = './npinter_sequences (2).csv'
    test_prot, test_rna, y_test = read_train_data(file_name=test_file)

    # Feature Extraction for TRAIN DATA

    train_features = []

    for i in range(0, len(train_rna)):
        # print(i)
        # PROTEIN Feature Extraction
        kmers = list(count_kmers(train_prot[i], k=5).keys())
        prot_feature_dict, msu_rep = feature_extract(
            kmers=kmers, mapping_type=prot_mapping, sample_type_features=prot_data_features)

        # RNA Feature Extraction
        kmers = list(count_kmers(train_rna[i], k=5).keys())
        rna_feature_dict, msu_rep = feature_extract(
            kmers=kmers, mapping_type=rna_mapping, sample_type_features=rna_data_features)

        sample_features = list(prot_feature_dict.values()) + list(rna_feature_dict.values())

        # <-PROTEIN FEATURE, APPEND THIS TO RNA FEATURE AND SAVE AS ONE LIST
        train_features.append(sample_features)

    X_train = train_features

    # Feature Extraction for TEST DATA

    test_features = []

    for i in range(0, len(test_rna)):
        # print(i)
        # PROTEIN Feature Extraction
        kmers = list(count_kmers(test_prot[i], k=5).keys())
        prot_feature_dict, msu_rep = feature_extract(
            kmers=kmers, mapping_type=prot_mapping, sample_type_features=prot_data_features)

        # RNA Feature Extraction
        kmers = list(count_kmers(test_rna[i], k=5).keys())
        rna_feature_dict, msu_rep = feature_extract(
            kmers=kmers, mapping_type=rna_mapping, sample_type_features=rna_data_features)

        sample_features = list(prot_feature_dict.values()) + \
                          list(rna_feature_dict.values())
        # print(sample_features)
        # <-PROTEIN FEATURE, APPEND THIS TO RNA FEATURE AND SAVE AS ONE LIST
        test_features.append(sample_features)

    X_test = test_features

    # print(X_test)

    # model

    params = {'objective': 'binary:logistic', 'n_estimators': 200, 'learning_rate': 0.25, 'max_depth': 8,
              'reg_alpha': 1.12,
              'reg_lambda': 18.51, 'subsample': 0.9}

    model = xgb.XGBClassifier(**params)

    # prepare the cross-validation procedure
    cv = KFold(n_splits=10, shuffle=True)

    # Making Scorer
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}

    # evaluate model
    scores = cross_validate(estimator=model, X=X_train, y=y_train, scoring=scoring, cv=cv, n_jobs=-1)

    # report performance
    for i in list(scores.keys())[2:]:
        mean_scores = np.mean(scores[i])
        std_scores = np.std(scores[i])

        print(i, ":", mean_scores, " ", std_scores)

    # report performance
    for i in list(scores.keys())[2:]:
        mean_scores = np.mean(scores[i])
        std_scores = np.std(scores[i])

        print(i, ":", mean_scores, " ", std_scores)

    # model fitting

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # saving the model
    pickle.dump(model, open('model.pkl', 'wb'))

    acc = accuracy_score(y_true=y_test, y_pred=preds)
    return acc


if __name__ == "__main__":
    accuracy = train_the_model_and_calculate_metrics()

    print("accuracy", accuracy)
