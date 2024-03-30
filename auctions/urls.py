from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("login", views.login_view, name="login"),
    path("logout", views.logout_view, name="logout"),
    path("register", views.register, name="register"),
    path("suicidalprediction", views.suicidalprediction, name="suicidalprediction"),
    path("breastcancerdetection", views.breastcancerdetection, name="breastcancerdetection"),
    path("fraudulentjob", views.fraudulentjob, name="fraudulentjob"),
    path("dogsvscats", views.dogsvscats, name="dogsvscats")
]
