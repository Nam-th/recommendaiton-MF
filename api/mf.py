# views.py
import pandas as pd
from django.http import JsonResponse
from rest_framework.views import APIView
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder

class RecommendationView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load and preprocess data
        data = pd.read_csv("train_data.csv", names=["UserID", "ProductID", "Rating"])
        
        # Encode UserID and ProductID to categorical integer values
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        data["UserID"] = self.user_encoder.fit_transform(data["UserID"])
        data["ProductID"] = self.product_encoder.fit_transform(data["ProductID"])
        
        # Create a utility matrix
        self.utility_matrix = data.pivot(index="UserID", columns="ProductID", values="Rating").fillna(0)
        
        # Train the MF model
        self.model = NMF(n_components=10, init="random", random_state=42)
        self.user_features = self.model.fit_transform(self.utility_matrix)
        self.item_features = self.model.components_

    def get_recommendations(self, user_id):
        user_idx = self.user_encoder.transform([user_id])[0]
        user_ratings = self.user_features[user_idx].dot(self.item_features)
        
        # Recommend products with highest predicted ratings
        recommended_products = (-user_ratings).argsort()[:5]  # Get top 5 recommendations
        product_ids = self.product_encoder.inverse_transform(recommended_products)
        
        return product_ids

    def get(self, request, user_id):
        recommendations = self.get_recommendations(user_id)
        return JsonResponse({"recommended_products": list(recommendations)})
