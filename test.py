from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import KNNWithMeans
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

app = Flask(__name__)

# Load your dataset
df = pd.read_csv('new_df.csv').iloc[:50000, :]

# Collaborative Filtering using KNN
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)
trainset, testset = train_test_split(data, test_size=0.3, random_state=10)
algo_collaborative = KNNWithMeans(k=20, sim_options={'name': 'cosine', 'user_based': False})
algo_collaborative.fit(trainset)

# Item-based Matrix Factorization
ratings_matrix = df.pivot_table(values='rating', index='username', columns='item_id', fill_value=0)
X = ratings_matrix.T
SVD = TruncatedSVD(n_components=20)
decomposed_matrix = SVD.fit_transform(X)
correlation_matrix = np.corrcoef(decomposed_matrix)

# Function to get product link from Excel
def get_product_link(item_id):
    df = pd.read_excel('E:/ngocquy_python/RCM/linksid.xlsx')
    rows = df[df['item_id'].isin(item_id)]
    
    product_links = []
    for _, row in rows.iterrows():
        product_links.append(row['product_link'])
    return product_links

def get_recommendations(item_id):
    # Collaborative Filtering
    test_pred = algo_collaborative.predict('user_id', item_id)
    collaborative_score = test_pred.est

    # Item-based Matrix Factorization
    product_names = list(X.index)
    product_ID = product_names.index(item_id)
    correlation_product_ID = correlation_matrix[product_ID]
    item_based_recommendations = list(X.index[correlation_product_ID > 0.6])
    item_based_recommendations.remove(item_id)

    # Combined Score
    combined_score = 0.7 * collaborative_score + 0.3 * len(item_based_recommendations)

    # Get product link
    product_link = get_product_link(item_based_recommendations[:10])

    return combined_score, item_based_recommendations[:10], product_link

@app.route('/')
def index():
    product_names = list(X.index)
    return render_template('index.html', product_names=product_names)


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        item_id = request.form['item_id']
        recommended_score, recommendations, product_link = get_recommendations(int(item_id))
        return render_template('recommendations.html', item_id=int(item_id), score=recommended_score,recommendations = recommendations, product_link=product_link)

if __name__ == '__main__':
    app.run(debug=True)
