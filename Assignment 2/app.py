from flask import Flask, render_template, jsonify, request, redirect, url_for
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from random import sample
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np 
import json
import pandas as pd

app = Flask(__name__)

# Global variables initialization

data_df = pd.read_csv("data/encoded_data.csv")
print("Whole dataset loaded into RAM")

records = {}

records["whole_dataset"] = []
records["stratified_sample"] = []
records["random_sample"] = []

records["data_loaded"] = False

# records["pca_loadings"] = {}
# records["pca_loadings"]["whole_dataset"] = None
# records["pca_loadings"]["stratified_sample"] = None
# records["pca_loadings"]["random_sample"] = None

records["cumulative_pca_variance_ratio"] = {}
# records["cumulative_pca_variance_ratio"]["whole_dataset"] = None
# records["cumulative_pca_variance_ratio"]["stratified_sample"] = None
# records["cumulative_pca_variance_ratio"]["random_sample"] = None

records["intrinsic_dimensionality"] = {}
# records["intrinsic_dimensionality"]["whole_dataset"] = 0
# records["intrinsic_dimensionality"]["stratified_sample"] = 0
# records["intrinsic_dimensionality"]["random_sample"] = 0

records["top_3_attributes"] = {}
# records["top_3_attributes"]["whole_dataset"] = None
# records["top_3_attributes"]["stratified_sample"] = None
# records["top_3_attributes"]["random_sample"] = None

saved_pca = {}
# saved_pca["whole_dataset"] = None
# saved_pca["stratified_sample"] = None
# saved_pca["random_sample"] = None


@app.route('/', methods = ["GET"])
def index():

    # global records

    # if not records["data_type"]:
    return render_template("index.html")
    # else:
    #     return render_template("index.html", data = records[records["data_type"]], pca = records["pca"][records["data_type"]], \
    #                                         pca_loadings = records["pca_loadings"][records["data_type"]])
    # if request.method == 'POST':
    #    query = request.form['query']
    #    print(query)
    #    di = {"name":"Aditya"}
    #    temp = [1,2,3,4,5]
    #    return render_template("index.html", temp = temp, di = di)
    # elif request.method == "GET":
    #     return render_template("index.html")

# @app.route('/csv', methods = ["GET", "POST"])
# def get_csv():
#     if request.method == 'POST':
#         sample = data_df.sample(frac=0.25, random_state=11)
#         return {"test": sample.to_dict("records")}
#     else:
#         return {"a": 2}

@app.route('/api/generate_data', methods = ["POST"])
def generate_data():

    global records
    
    scaler = MinMaxScaler()

    if not records["data_loaded"]:

        records["data_loaded"] = True

        """
        Stratified sampling code

        """
        print("First stratified call")

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

        saved_pca["stratified_sample"] = pca

        for feature, pca_variance, pca_variance_ratio in zip(list(range(pca.n_components_)), list(pca.explained_variance_),\
                                                                list(pca.explained_variance_ratio_)):
            records["stratified_sample"].append({"feature":feature, "pca_variance":pca_variance, \
                                                    "pca_variance_ratio":pca_variance_ratio})

        # makes it easier to get 75% data variance
        records["stratified_sample"] = sorted(records["stratified_sample"], key = lambda x: -x["pca_variance"])

        # calculating cumulative ratio and intrinsic dimensionality
        calculate_cum_sum("stratified_sample")
        get_loaded_attributes("stratified_sample")

        # records["pca_loadings"]["stratified_sample"] = pd.DataFrame(data = pca.components_, columns = ["pca_" + str(i) for i in range(pca.n_components_)]).to_dict("records")
    
        """
        Random sampling code

        """
        print("First random call")
        random_sample = data_df.sample(frac=0.25, random_state=11)

        pca = PCA(n_components=15)

        pca.fit(scaler.fit_transform(random_sample))

        saved_pca["random_sample"] = pca

        for feature, pca_variance, pca_variance_ratio in zip(list(range(pca.n_components_)), list(pca.explained_variance_),\
                                                    list(pca.explained_variance_ratio_)):
            records["random_sample"].append({"feature":feature, "pca_variance":pca_variance, \
                                                    "pca_variance_ratio":pca_variance_ratio})

        # makes it easier to get 75% data variance
        records["random_sample"] = sorted(records["random_sample"], key = lambda x: -x["pca_variance"])

        # calculating cumulative ratio and intrinsic dimensionality
        calculate_cum_sum("random_sample")
        get_loaded_attributes("random_sample")

        # records["pca_loadings"]["random_sample"] = pd.DataFrame(data = pca.components_, columns = ["pca_" + str(i) for i in range(pca.n_components_)]).to_dict("records")
        # records["pca"]["whole_dataset"] = pca.fit_transform()
    
        """
        PCA for the whole dataset

        """
        pca = PCA(n_components=15)

        pca.fit(scaler.fit_transform(data_df))

        saved_pca["whole_dataset"] = pca

        for feature, pca_variance, pca_variance_ratio in zip(list(range(pca.n_components_)), list(pca.explained_variance_),\
                                                    list(pca.explained_variance_ratio_)):
            records["whole_dataset"].append({"feature":feature, "pca_variance":pca_variance, \
                                                    "pca_variance_ratio":pca_variance_ratio})

        # makes it easier to get 75% data variance
        records["whole_dataset"] = sorted(records["whole_dataset"], key = lambda x: -x["pca_variance"])

        # calculating cumulative ratio and intrinsic dimensionality
        calculate_cum_sum("whole_dataset")
        get_loaded_attributes("whole_dataset")

        # records["pca_loadings"]["whole_dataset"] = pd.DataFrame(data = pca.components_, columns = ["pca_" + str(i) for i in range(pca.n_components_)]).to_dict("records")
        # records["pca"]["whole_dataset"] = pca.fit_transform()

    return json.dumps(records)

def calculate_cum_sum(data_type):
    records["cumulative_pca_variance_ratio"][data_type] = []

    for i in range(len(records[data_type])):
            records["cumulative_pca_variance_ratio"][data_type].append({"feature": records[data_type][i]["feature"], \
                                                                        "pca_variance_ratio": records[data_type][i]["pca_variance_ratio"]})
    
    records["intrinsic_dimensionality"][data_type] = 0
    for i in range(len(records[data_type])):
        records["cumulative_pca_variance_ratio"][data_type][i]["pca_variance_ratio"] += records["cumulative_pca_variance_ratio"][data_type][i - 1]["pca_variance_ratio"]
        
        if records["cumulative_pca_variance_ratio"][data_type][i]["pca_variance_ratio"] <= 0.75:
            records["intrinsic_dimensionality"][data_type] += 1

def get_loaded_attributes(data_type):
    
    pca_principal_axes = []

    for i in range(records["intrinsic_dimensionality"][data_type]):
        pca_principal_axes.append(records[data_type][i]["feature"])
    
    # print(pca_principal_axes)

    pca = saved_pca[data_type]

    # rows represent the variables, columns represent number of pca_components (only principal axes)
    loadings = pca.components_[np.ix_(pca_principal_axes)].transpose()

    sum_of_square_values = np.square(loadings).sum(axis = 1)

    feature_value = []

    for squared_sum, feature in zip(sum_of_square_values, data_df.columns):
        feature_value.append({"variable": feature, "squared_value": squared_sum})

    feature_value.sort(key = lambda x: -x["squared_value"])

    records["top_3_attributes"][data_type] = feature_value


if __name__ == '__main__':
    app.run(debug=True)