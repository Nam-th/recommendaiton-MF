from pymongo import MongoClient
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import os
from django.http import JsonResponse
from rest_framework.views import APIView
import numpy as np

class RecommendationView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Kết nối MongoDB
        self.client = MongoClient(
            "mongodb+srv://datvan635:GXnsBWtxBCFR9I93@book-wise.scywq.mongodb.net/book-wise?retryWrites=true&w=majority"
        )
        self.db = self.client["book-wise"]
        self.ratings_collection = self.db["reviews"]
        self.books_collection = self.db["books"]
        
        # File lưu trữ ma trận và encoder
        self.model_path = "nmf_model_data"
        self.user_features_file = os.path.join(self.model_path, "user_features.npy")
        self.item_features_file = os.path.join(self.model_path, "item_features.npy")
        self.user_encoder_file = os.path.join(self.model_path, "user_encoder.pkl")
        self.product_encoder_file = os.path.join(self.model_path, "product_encoder.pkl")

        # Load dữ liệu huấn luyện nếu đã có sẵn, nếu không thì train
        if os.path.exists(self.user_features_file):
            self.load_model()
        else:
            self.train_model()

        # Luôn khởi tạo self.book_data từ MongoDB
        books_data = list(self.books_collection.find({}, {"_id": 0}))
        self.book_data = pd.DataFrame(books_data)

    def train_model(self):
        # Lấy dữ liệu ratings từ MongoDB
        ratings_data = list(self.ratings_collection.find({"is_deleted": False}, {"_id": 0}))
        if not ratings_data:
            raise ValueError("Không có dữ liệu ratings trong MongoDB để huấn luyện!")

        data = pd.DataFrame(ratings_data).rename(columns={"book_id": "book_id"})
        data = data[data["rating"].notnull()]
        data = data[data["rating"].apply(lambda x: isinstance(x, (int, float)))]
        data["rating"] = data["rating"].astype(float)
        data = data.groupby(["user_id", "book_id"], as_index=False)["rating"].mean()

        # Mã hóa user_id và book_id
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        data["user_id"] = self.user_encoder.fit_transform(data["user_id"])
        data["book_id"] = self.product_encoder.fit_transform(data["book_id"])

        # Tạo utility matrix
        utility_matrix = data.pivot(index="user_id", columns="book_id", values="rating").fillna(0)

        # Huấn luyện mô hình
        model = NMF(n_components=15, init="nndsvd", max_iter=500, random_state=42)
        self.user_features = model.fit_transform(utility_matrix)
        self.item_features = model.components_

        # Lưu dữ liệu
        os.makedirs(self.model_path, exist_ok=True)
        np.save(self.user_features_file, self.user_features)
        np.save(self.item_features_file, self.item_features)
        with open(self.user_encoder_file, "wb") as f:
            pickle.dump(self.user_encoder, f)
        with open(self.product_encoder_file, "wb") as f:
            pickle.dump(self.product_encoder, f)

    def load_model(self):
        # Tải dữ liệu đã lưu
        self.user_features = np.load(self.user_features_file)
        self.item_features = np.load(self.item_features_file)
        with open(self.user_encoder_file, "rb") as f:
            self.user_encoder = pickle.load(f)
        with open(self.product_encoder_file, "rb") as f:
            self.product_encoder = pickle.load(f)

    def get_recommendations(self, user_id):
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
        except ValueError:
            return []

        user_ratings = self.user_features[user_idx].dot(self.item_features)
        recommended_products = (-user_ratings).argsort()[:20]
        product_ids = self.product_encoder.inverse_transform(recommended_products)
        recommended_books = self.book_data[self.book_data["book_id"].isin(product_ids)]
        return recommended_books.to_dict(orient="records")

    def get(self, request, user_id):
        recommendations = self.get_recommendations(user_id)
        return JsonResponse({"recommended_books": recommendations})
