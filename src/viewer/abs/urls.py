from django.urls import path
from abs.views import abs_index_view, abs_headlines

urlpatterns = [
    path("", abs_index_view, name="abs-homepage"),
    path("headlines", abs_headlines, name="abs-headlines"),
]
