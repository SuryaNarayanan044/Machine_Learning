from django.urls import path
from .import views
urlpatterns=[
    path('index',views.index,name='index'),
    path('signup',views.signup,name='signup'),
    path('login',views.login,name='login'),
    path('logout',views.logout,name='logout'),
    path('recommendation',views.recommendation,name='recommendation'),
    path('crop-recommend',views.crop_recommendation,name='crop-recommend'),
    path('fertilizer-recommend',views.fertilizer_recommendation,name='fertilizer-recommend'),
    
]