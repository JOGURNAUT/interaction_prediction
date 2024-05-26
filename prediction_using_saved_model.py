import pickle
from model_creation_and_saving import count_kmers, feature_extract, prot_mapping, rna_mapping, prot_data_features, \
    rna_data_features

pickled_model = pickle.load(open('model.pkl', 'rb'))
# pickled_model.predict(X_test)

protein_feature_input = []
rna_feature_input = []
protein_feature_var = input("Enter protein feature :")
print(protein_feature_var)
protein_feature_input.append(protein_feature_var)
rna_feature_var = input("Enter rna feature : ")
print(rna_feature_var)
rna_feature_input.append(rna_feature_var)


def predict_y_val(test_prot, test_rna):
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
    preds = pickled_model.predict_proba(X_test)[:,-1]
    return preds


if __name__ == "__main__":
    predict_ion = predict_y_val(protein_feature_input, rna_feature_input)
    print("prediction of input", predict_ion)
