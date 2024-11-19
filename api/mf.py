from pymongo import MongoClient
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

class Recommender:
    def __init__(self, db_uri, db_name):
        client = MongoClient(db_uri)
        db = client[db_name]
        self.ratings_collection = db["reviews"]
        self.books_collection = db["books"]
        self.load_data()
        self.load_model()

    def load_data(self):
        # Lấy dữ liệu và xử lý
        ratings_data = list(self.ratings_collection.find({"is_deleted": False}, {"_id": 0}))
        data = pd.DataFrame(ratings_data).rename(columns={"book_id": "book_id"})
        data["rating"] = data["rating"].astype(float)
        data = data.groupby(["user_id", "book_id"], as_index=False)["rating"].mean()
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        data["user_id"] = self.user_encoder.fit_transform(data["user_id"])
        data["book_id"] = self.product_encoder.fit_transform(data["book_id"])
        self.utility_matrix = data.pivot(index="user_id", columns="book_id", values="rating").fillna(0)

    def train_model(self):
        # Huấn luyện mô hình
        self.model = NMF(n_components=15, init="nndsvd", max_iter=500, random_state=42)
        self.user_features = self.model.fit_transform(self.utility_matrix)
        self.item_features = self.model.components_

    def save_model(self, filepath="nmf_model.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump((self.user_features, self.item_features), f)

    def load_model(self, filepath="nmf_model.pkl"):
        with open(filepath, "rb") as f:
            self.user_features, self.item_features = pickle.load(f)

    def recommend(self, user_id):
        # Logic gợi ý
        user_idx = self.user_encoder.transform([user_id])[0]
        user_ratings = self.user_features[user_idx].dot(self.item_features)
        recommended_products = (-user_ratings).argsort()[:20]
        product_ids = self.product_encoder.inverse_transform(recommended_products)
        books = list(self.books_collection.find({"book_id": {"$in": product_ids}}, {"_id": 0}))
        return books
