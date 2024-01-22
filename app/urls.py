from django.contrib import admin
from django.urls import path, include
from .views import *

urlpatterns = [
    path("bot/", bot, name="bot"),
    path('dashboard/',dashboard, name='dashboard'),
    path('login/', login_view, name='login'),
    path('signup/', signup_view, name='signup'),
    path('logout/', logout_view, name='logout'),
    path('',login_view, name='login'),
    path('report_nlp/',report_nlp, name='nlp_report'),
    path('sentiment',sentiment_analysis,name='sentiment'),
    path('cctv',cctv,name='cctv'),
    path('health_report',mood_analysis_view,name='health_report'),
    path('prioritize_tasks/', prioritize_tasks, name='prioritize_tasks'),
    path('prioritize_tasks_from_db/', prioritize_tasks_from_db, name='prioritize_tasks_from_db'),
   
    
]
