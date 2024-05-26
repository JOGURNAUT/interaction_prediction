import streamlit as st
import numpy as np
import pickle
from model_creation_and_saving import count_kmers, feature_extract, prot_mapping, rna_mapping, prot_data_features, \
    rna_data_features

pickled_model = pickle.load(open('model.pkl', 'rb'))

# Load the trained ML model (you need to replace 'model.pkl' with your actual model file)
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)


# Streamlit UI
st.title('Protein-RNA Interaction Prediction')

st.header('Input Features')
protein_feature = st.text_input('Protein Feature', max_chars=1000)
rna_feature = st.text_input('RNA Feature', max_chars=1000)

# Prediction
if st.button('Predict Interaction'):
    #features = np.array([[protein_feature, rna_feature]])
    test_prot = []
    test_rna = []
    test_prot.append(protein_feature)
    test_rna.append(rna_feature)
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
    prediction = pickled_model.predict_proba(X_test)[:,-1]

    if prediction >= 0.5:
        st.success('The sequences are more likely to interact.')
    else:
        st.error('The sequences are less likely to interact.')
