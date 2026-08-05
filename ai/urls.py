from django.urls import path

from ai import views

urlpatterns = [
    path("", views.ai_main, name="ai_main"),
    path("run_chat/", views.run_chat, name="run_chat"),
]
