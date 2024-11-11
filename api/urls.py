from django.urls import path
from .views import get_users, create_user, user_detail, RecommendationView

urlpatterns = [
    path('users/', get_users, name='get_users'),
    path('users/create/', create_user, name='create_user'),
    path('users/<int:pk>', user_detail, name='user_detail'),
    path('recommendations/<str:user_id>', RecommendationView.as_view(), name='recommend'),
]