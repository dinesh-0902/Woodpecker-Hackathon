from django.urls import path
from .views import predict_rainfall

urlpatterns = [
    path('predict/', predict_rainfall, name='predict_rainfall'),
]
