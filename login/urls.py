from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.home, name="home"),
    path('about/', views.about, name="about"),
    
    path('register/', views.register, name="register"),
    path('signin/', views.signin, name="signin"),
    path('signout/', views.signout, name="signout"),
    
    path('admin_home/', views.admin_home, name="admin_home"),
    path('staff_home/', views.staff_home, name="staff_home"),

    path('profile/', views.profile, name="profile"),
    path('edit_profile/', views.edit_profile, name="edit_profile"),

    path('hospital_staff_authorize/', views.hospital_staff_authorize, name="hospital_staff_authorize"),
    path('accept/<int:id>/', views.accept, name="accept"),
    path('reject/<int:id>/', views.reject, name="reject"),
    
    path('add_feedback/', views.add_feedback, name='add_feedback'),
    path('feedback/', views.feedback, name='feedback'),

    path('view_patient/', views.view_patient, name='view_patient'),
    path('add_patient/', views.add_patient, name='add_patient'),
    path('edit_patient/<int:id>/', views.edit_patient, name='edit_patient'),
    path('delete_patient/<int:id>/', views.delete_patient, name='delete_patient'),

    path("icd/ajax/", views.icd_ajax_predict, name="icd_ajax_predict"),
    path("ping/", views.ping, name="ping"),


]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)