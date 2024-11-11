from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
import pandas as pd
import os
from django.http import JsonResponse
from .models import User
from .serializer import UserSerializer
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder

@api_view(['GET'])
def get_users(request):
    users = User.objects.all()
    serializer = UserSerializer(users, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def create_user(request): 
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid(): 
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT', 'DELETE'])
def user_detail (request, pk):
    try:
        user = User.objects.get(pk=pk)
    except User.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    if request.method == 'GET':
        serializer = UserSerializer(user)
        return Response(serializer.data)
    
class RecommendationView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Đường dẫn tới file train_data.csv
        data_path = os.path.join(os.path.dirname(__file__), "train_data.csv")
        
        # Load dữ liệu và gán tên cột
        data = pd.read_csv(data_path, header=None, names=["UserID", "ProductID", "Rating"])
        
        # Xử lý các bản ghi trùng lặp bằng cách tính trung bình
        data = data.groupby(["UserID", "ProductID"], as_index=False).mean()
        
        # Mã hóa UserID và ProductID thành các giá trị số
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        data["UserID"] = self.user_encoder.fit_transform(data["UserID"])
        data["ProductID"] = self.product_encoder.fit_transform(data["ProductID"])
        
        # Tạo utility matrix
        self.utility_matrix = data.pivot(index="UserID", columns="ProductID", values="Rating").fillna(0)
        
        # Huấn luyện mô hình MF (NMF)
        self.model = NMF(n_components=10, init="random", random_state=42)
        self.user_features = self.model.fit_transform(self.utility_matrix)
        self.item_features = self.model.components_

    def get_recommendations(self, user_id):
        # Chuyển UserID từ dạng mã hoá sang dạng chỉ mục của ma trận
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
        except ValueError:
            return []  # Trả về danh sách rỗng nếu user_id không tồn tại

        # Dự đoán xếp hạng cho tất cả sản phẩm
        user_ratings = self.user_features[user_idx].dot(self.item_features)
        
        # Lấy các sản phẩm có xếp hạng dự đoán cao nhất
        recommended_products = (-user_ratings).argsort()[:20]  # Lấy 5 gợi ý hàng đầu
        product_ids = self.product_encoder.inverse_transform(recommended_products)
        
        # Chuyển đổi product_ids thành kiểu int để JSON serializable
        return [int(product_id) for product_id in product_ids]

    def get(self, request, user_id):
        # Gọi hàm get_recommendations để lấy danh sách gợi ý
        recommendations = self.get_recommendations(int(user_id))
        return JsonResponse({"recommended_products": recommendations})