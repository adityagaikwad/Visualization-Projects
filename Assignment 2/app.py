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
records = {}

records["data_df"] = pd.read_csv("data/encoded_data.csv")
records["whole_dataset"] = []
records["stratified_sample"] = []
records["random_sample"] = []
records["data_type"] = None

print("Whole dataset loaded into RAM")


@app.route('/', methods = ["GET"])
def index():

    global records

    if not records["data_type"]:
        return render_template("index.html")
    else:
        return render_template("index.html", data = records[records["data_type"]])
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

@app.route('/api/<data_type>', methods = ["GET"])
def generate_data(data_type):

    global records
    
    scaler = MinMaxScaler()

    n_components = 15

    if data_type == "stratified":

        # if sample made already, don't run code again
        if not records["stratified_sample"]:
            print("First stratified call")

            optimal_k = 7

            kmeanModel = KMeans(n_clusters=optimal_k, random_state=11)
            kmeanModel.fit(records["data_df"])

            clusters = {i:[] for i in range(optimal_k)}

            cluster_labels = kmeanModel.labels_

            for record, cluster_number in zip(records["data_df"].iterrows(), cluster_labels):
                clusters[cluster_number].append(record[1])

            percent = 0.25
            stratified_sample = []

            for cluster_number in clusters.keys():
                stratified_sample.extend(sample(clusters[cluster_number], int(len(clusters[cluster_number]) * percent)))

            pca = PCA(n_components=n_components)

            pca.fit(scaler.fit_transform(stratified_sample))
            pca_output = pd.DataFrame(data = pca.fit_transform(scaler.fit_transform(stratified_sample)), columns = ["pca_" + str(i) for i in range(n_components)])
            records["stratified_sample"].append(pca_output)

            for feature, pca_variance, pca_variance_ratio in zip(list(range(pca.n_components_)), list(pca.explained_variance_),\
                                                                 list(pca.explained_variance_ratio_)):
                records["stratified_sample"].append({"feature":feature, "pca_variance":pca_variance, \
                                                     "pca_variance_ratio":pca_variance_ratio})

            # makes it easier to get 75% data variance
            records["stratified_sample"] = [list(records["stratified_sample"][0].to_dict("records"))] + sorted(records["stratified_sample"][1:], key = lambda x: -x["pca_variance"])

        records["data_type"] = "stratified_sample"
    
    elif data_type == "random":

        if not records["random_sample"]:
            print("First random call")
            random_sample = records["data_df"].sample(frac=0.25, random_state=11)

            pca = PCA(n_components=n_components)

            pca.fit(scaler.fit_transform(random_sample))
            pca_output = pd.DataFrame(data = pca.fit_transform(scaler.fit_transform(random_sample)), columns = ["pca_" + str(i) for i in range(n_components)])

            records["random_sample"].append(pca_output)
            
            for feature, pca_variance, pca_variance_ratio in zip(list(range(pca.n_components_)), list(pca.explained_variance_),\
                                                        list(pca.explained_variance_ratio_)):
                records["random_sample"].append({"feature":feature, "pca_variance":pca_variance, \
                                                     "pca_variance_ratio":pca_variance_ratio})

            # makes it easier to get 75% data variance
            records["random_sample"] = [list(records["random_sample"][0].to_dict("records"))] + sorted(records["random_sample"][1:], key = lambda x: -x["pca_variance"])

        records["data_type"] = "random_sample"
    
    elif data_type == "whole":

        if not records["whole_dataset"]:
            pca = PCA(n_components=n_components)

            pca.fit(scaler.fit_transform(records["data_df"]))
            pca_output = pd.DataFrame(data = pca.fit_transform(scaler.fit_transform(records["data_df"])), columns = ["pca_" + str(i) for i in range(n_components)])
            records["whole_dataset"].append(pca_output)

            for feature, pca_variance, pca_variance_ratio in zip(list(range(pca.n_components_)), list(pca.explained_variance_),\
                                                        list(pca.explained_variance_ratio_)):
                records["whole_dataset"].append({"feature":feature, "pca_variance":pca_variance, \
                                                     "pca_variance_ratio":pca_variance_ratio})

            # makes it easier to get 75% data variance
            records["whole_dataset"] = [list(records["whole_dataset"][0].to_dict("records"))] + sorted(records["whole_dataset"][1:], key = lambda x: -x["pca_variance"])

        records["data_type"] = "whole_dataset"

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)