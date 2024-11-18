from pymongo import MongoClient
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from django.http import JsonResponse
from rest_framework.views import APIView

class RecommendationView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Kết nối tới MongoDB Atlas
        client = MongoClient(
            "mongodb+srv://datvan635:GXnsBWtxBCFR9I93@book-wise.scywq.mongodb.net/book-wise?retryWrites=true&w=majority"
        )
        
        # Lấy database và collection
        db = client["book-wise"]  # Tên database
        ratings_collection = db["reviews"]  # Collection lưu ratings
        books_collection = db["books"]  # Collection lưu thông tin sách
        
        # Lấy dữ liệu ratings từ MongoDB
        ratings_data = list(ratings_collection.find({"is_deleted": False}, {"_id": 0}))
        if not ratings_data:
            raise ValueError("Không có dữ liệu ratings trong MongoDB để huấn luyện!")
        
        # Chuyển dữ liệu thành DataFrame
        data = pd.DataFrame(ratings_data).rename(columns={"book_id": "book_id"})
        data = data[data["rating"].notnull()]  # Loại bỏ giá trị null
        data = data[data["rating"].apply(lambda x: isinstance(x, (int, float)))]  # Chỉ giữ giá trị số

        # Chuyển đổi rating sang kiểu float để đảm bảo số thực có thể được xử lý
        data["rating"] = data["rating"].astype(float)
        # Xử lý các bản ghi trùng lặp bằng cách tính trung bình
        data = data.groupby(["user_id", "book_id"], as_index=False)["rating"].mean()

        # Mã hóa user_id và book_id thành các giá trị số
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        data["user_id"] = self.user_encoder.fit_transform(data["user_id"])
        data["book_id"] = self.product_encoder.fit_transform(data["book_id"])
        
        # Tạo utility matrix
        self.utility_matrix = data.pivot(index="user_id", columns="book_id", values="rating").fillna(0)
        
        # Huấn luyện mô hình MF (NMF)
        # Huấn luyện mô hình MF (NMF)
        self.model = NMF(
        n_components=15,         # Bắt đầu với 10 latent factors
        init="nndsvd",           # Khởi tạo thông minh
        max_iter=500,            # Số vòng lặp tối đa
        random_state=42,         # Đảm bảo tái lập
        beta_loss="frobenius",   # Hàm mất mát mặc định
        ) 
        self.user_features = self.model.fit_transform(self.utility_matrix)
        self.item_features = self.model.components_
        
        # Lấy thông tin chi tiết sách từ MongoDB
        books_data = list(books_collection.find({}, {"_id": 0}))
        self.book_data = pd.DataFrame(books_data)

    def get_recommendations(self, user_id):
        # Chuyển user_id từ dạng mã hoá sang dạng chỉ mục của ma trận
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
        except ValueError:
            return []  # Trả về danh sách rỗng nếu user_id không tồn tại

        # Dự đoán xếp hạng cho tất cả sản phẩm
        user_ratings = self.user_features[user_idx].dot(self.item_features)
        
        # Lấy các sản phẩm có xếp hạng dự đoán cao nhất
        recommended_products = (-user_ratings).argsort()[:20]  # Lấy 20 gợi ý hàng đầu
        product_ids = self.product_encoder.inverse_transform(recommended_products)
        
        # Lọc thông tin sách từ `book_data`
        recommended_books = self.book_data[self.book_data["book_id"].isin(product_ids)]
        return recommended_books.to_dict(orient="records")

    def get(self, request, user_id):
        # Gọi hàm get_recommendations để lấy danh sách gợi ý
        recommendations = self.get_recommendations((user_id))
        return JsonResponse({"recommended_books": recommendations})
