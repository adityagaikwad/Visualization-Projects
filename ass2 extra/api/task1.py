from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from random import sample
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np 
import json
import pandas as pd

data_df = pd.read_csv("data/encoded_data.csv")
print("Whole dataset loaded into RAM")

def sample_data(data_type):

    scaler = MinMaxScaler()

    if data_type == "stratified":

        optimal_k = 7

        kmeanModel = KMeans(n_clusters=optimal_k, random_state=11)
        kmeanModel.fit(data_df)

        clusters = {i:[] for i in range(optimal_k)}

        cluster_labels = kmeanModel.labels_

        for record, cluster_number in zip(data_df.iterrows(), cluster_labels):
            clusters[cluster_number].append(record[1])

        percent = 0.25
        stratified_sample = []

        for cluster_number in clusters.keys():
            stratified_sample.extend(sample(clusters[cluster_number], int(len(clusters[cluster_number]) * percent)))

        pca = PCA(n_components=15)

        pca.fit(scaler.fit_transform(stratified_sample))

        stratified_sample = []

        for feature, pca_variance, pca_variance_ratio in zip(list(range(pca.n_components_)), list(pca.explained_variance_),\
                                                                list(pca.explained_variance_ratio_)):
            stratified_sample.append({"feature":feature, "pca_variance":pca_variance, \
                                                    "pca_variance_ratio":pca_variance_ratio})

        # makes it easier to get 75% data variance
        stratified_sample = sorted(stratified_sample, key = lambda x: -x["pca_variance"])

        # records["pca_loadings"]["stratified_sample"] = pd.DataFrame(data = pca.components_, columns = ["pca_" + str(i) for i in range(pca.n_components_)]).to_dict("records")
        # records["pca"]["whole_dataset"] = pca.fit_transform()

        return json.dumps(stratified_sample)
        
    elif data_type == "random":

        random_sample = data_df.sample(frac=0.25, random_state=11)

        pca = PCA(n_components=15)

        pca.fit(scaler.fit_transform(random_sample))

        random_sample = []

        for feature, pca_variance, pca_variance_ratio in zip(list(range(pca.n_components_)), list(pca.explained_variance_),\
                                                    list(pca.explained_variance_ratio_)):
            random_sample.append({"feature":feature, "pca_variance":pca_variance, \
                                                    "pca_variance_ratio":pca_variance_ratio})

        # makes it easier to get 75% data variance
        random_sample = sorted(random_sample, key = lambda x: -x["pca_variance"])
        # records["pca_loadings"]["random_sample"] = pd.DataFrame(data = pca.components_, columns = ["pca_" + str(i) for i in range(pca.n_components_)]).to_dict("records")
        # records["pca"]["whole_dataset"] = pca.fit_transform()

        print(len(random_sample))
        return json.dumps(random_sample)
    
    elif data_type == "whole":

        # if not records["whole_dataset"]:
        pca = PCA(n_components=15)

        pca.fit(scaler.fit_transform(data_df))

        whole_dataset = []

        for feature, pca_variance, pca_variance_ratio in zip(list(range(pca.n_components_)), list(pca.explained_variance_),\
                                                    list(pca.explained_variance_ratio_)):
            whole_dataset.append({"feature":feature, "pca_variance":pca_variance, \
                                                    "pca_variance_ratio":pca_variance_ratio})

        print(len(whole_dataset))
        # makes it easier to get 75% data variance
        whole_dataset = sorted(whole_dataset, key = lambda x: -x["pca_variance"])

        # records["pca_loadings"]["whole_dataset"] = pd.DataFrame(data = pca.components_, columns = ["pca_" + str(i) for i in range(pca.n_components_)]).to_dict("records")
        # records["pca"]["whole_dataset"] = pca.fit_transform()

        return json.dumps(whole_dataset)