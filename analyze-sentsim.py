import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import faiss
import sys

df_train = pd.read_parquet('output/train_emb.parquet')
df_test = pd.read_parquet('output/test_emb.parquet')
print(df_train.sample(5))

feature_sizes = [5, 10, 20, 30, 40, 50, 75, 100, 250, 500]
emb_col = "emb_large"
target = "label"
results = []

def build_and_search_index(feature_size, data, batch_size):
    index = faiss.IndexFlatL2(feature_size)
    index.add(data)
    sim_results = []
    # for each record in the test set, determine whether record[0] is retrieved from record[1] and vice versa
    for i in range(0, len(df_test), batch_size):
        # check if the second sentence is retrieved from the first sentence
        # ask for 2 nearest neighbors because the first one is the query itself
        dists, indices = index.search(data[i:i + batch_size], 2)
        for j in range(len(indices)):
            # check that second nearest neighbor is the same as the index of the second sentence
            # (first nearest neighbor is the query itself)
            if indices[j][1] == len(df_test) + i + j:
                sim_results.append(1)
            else:
                sim_results.append(0)
        dists, indices = index.search(data[i+len(df_test):i+len(df_test) + batch_size], 2)
        for j in range(len(indices)):
            if indices[j][1] == i + j:
                sim_results.append(1)
            else:
                sim_results.append(0)
    return sim_results

for feature_size in feature_sizes:
    print(f"Evaluating feature size: {feature_size}")
    # first of sentence pairs have indexes 0-len(df_test), second of sentence pairs have indexes len(df_test)-2*len(df_test)
    normed = np.concatenate((np.array([x[0][:feature_size] for x in df_test[emb_col]]),
                             np.array([x[1][:feature_size] for x in df_test[emb_col]])), axis=0)
    sim_results = build_and_search_index(feature_size, normed, 128)
    print(f"SentSim accuracy on original embeddings: {sum(sim_results) / len(sim_results)}")
    results.append(("Truncation", feature_size, sum(sim_results) / len(sim_results)))

    pca_training_data = np.concatenate((np.array([x[0] for x in df_train[emb_col]]), np.array([x[1] for x in df_train[emb_col]])), axis=0)
    pca = PCA(n_components=feature_size)
    pca.fit(pca_training_data)
    pca_testing_data = np.concatenate((np.array([x[0] for x in df_test[emb_col]]), np.array([x[1] for x in df_test[emb_col]])), axis=0)
    pca_test_transformed = pca.transform(pca_testing_data)
    sim_results = build_and_search_index(feature_size, pca_test_transformed, 128)
    print(f"SentSim accuracy on PCA embeddings: {sum(sim_results) / len(sim_results)}")
    results.append(("PCA", feature_size, sum(sim_results) / len(sim_results)))

graph_df = pd.DataFrame({
    "method": [method for method, _, _ in results],
    "feature_size": [feature_size for _, feature_size, _ in results],
    "accuracy": [accuracy for _, _, accuracy in results]
})
#graph_df_pca = graph_df[graph_df["method"] == "PCA"]
#graph_df_truncation = graph_df[graph_df["method"] == "Truncation"]

sns.set_theme()
#sns.lineplot(data=graph_df_pca, x="feature_size", y="accuracy", label="PCA")
#sns.lineplot(data=graph_df_truncation, x="feature_size", y="accuracy", label="Truncation")
sns.barplot(data=graph_df, x="feature_size", y="accuracy", hue="method")
plt.legend(title="Method")
plt.xlabel("Feature Count")
plt.ylabel("Accuracy")
plt.title("Sentences Similarity Classification Accuracy")
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
plt.savefig("output/sentsim_accuracy.png")

