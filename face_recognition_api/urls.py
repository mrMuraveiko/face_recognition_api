from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from storage import views

urlpatterns = [
    path('api/v1/storage/', views.storage, name='storage'),
    path('api/v1/recognize/', views.recognize, name='recognize'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
