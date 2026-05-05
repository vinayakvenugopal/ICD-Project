from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.hashers import make_password
from .models import *
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from django.contrib.auth import update_session_auth_hash

# Create your views here.


def home(request):
    if request.user.is_authenticated:
        if request.user.is_superuser:
            return redirect('admin_home')
        return redirect('staff_home')
    return render(request, "index.html")


def admin_home(request):
    if not request.user.is_superuser:
        return redirect('home')
    total_patients = Patient.objects.count()
    pending_staff = User.objects.filter(status="pending").count()
    total_feedback = Feedback.objects.count()
    
    context = {
        'total_patients': total_patients,
        'pending_staff': pending_staff,
        'total_feedback': total_feedback,
    }
    return render(request, "admin_home.html", context)


def staff_home(request):
    if not request.user.is_authenticated or request.user.is_superuser:
        return redirect('home')
    total_patients = Patient.objects.count()
    
    context = {
        'total_patients': total_patients,
    }
    return render(request, "staff_home.html", context)


def register(request):
    if request.method == 'POST':
        profile = request.FILES.get('profile')
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        dob = request.POST.get('dob')
        gender = request.POST.get('gender')
        id_proof = request.FILES.get('id_proof')
        password = request.POST.get('password')

        role = "staff"

        if User.objects.filter(email=email).exists():
            messages.error(request, "email already exits")
            return redirect('register')
        
        result = User.objects.create(fullname=name, email=email, phone=phone, profile=profile, status="pending", id_proof=id_proof,
                                      gender=gender, dob=dob, role=role, password=make_password(password))
        result.save()
        return redirect('signin')
    return render(request, 'register.html')


def signin(request):
    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(request, email=email, password=password)

        if user is not None:
            # ✅ Superuser: allow login without status check
            if user.is_superuser:
                login(request, user)
                messages.success(request, "Admin login successful")
                return redirect('admin_home')

            # ✅ Normal user: check status
            elif user.status == "accepted":
                login(request, user)
                messages.success(request, "Login successful")
                return redirect('staff_home')

            else:
                messages.error(request, "Your account is pending approval")
                return redirect('signin')

        else:
            messages.error(request, "Invalid email or password")
            return redirect('signin')

    return render(request, 'login.html')


def signout(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect('home')


def profile(request):
    user = request.user
    return render(request, 'profile.html', {'user': user})


def edit_profile(request):
    user = request.user

    if request.method == 'POST':
        profile = request.FILES.get('profile')
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        dob = request.POST.get('dob')
        gender = request.POST.get('gender')
        id_proof = request.FILES.get('id_proof')
        password = request.POST.get('password')

        # Check if email already exists for another user
        if email and email != user.email:
            if User.objects.filter(email=email).exclude(id=user.id).exists():
                messages.error(request, "Email is already taken by another user.")
                return redirect('edit_profile')

        # Update profile image
        if profile:
            user.profile = profile

        if id_proof:
            user.id_proof = id_proof

        if dob:
            user.dob = dob
        else:
            user.dob = None

        if password:
            user.set_password(password)
            user.save()
            update_session_auth_hash(request, user)


        # Update other fields
        user.fullname = name
        user.email = email
        user.phone = phone
        user.gender = gender
        user.save()

        messages.success(request, "Profile updated successfully.")
        return redirect('profile')

    return render(request, 'edit_profile.html', {'user':user})


def hospital_staff_authorize(request):
    result = User.objects.filter(status="pending")
    return render(request, 'hospital_staff_authorize.html', {'result': result})


def accept(request, id):
    res = get_object_or_404(User, id=id)
    res.status = "accepted"
    res.save()
    return redirect('hospital_staff_authorize')


def reject(request, id):
    res = get_object_or_404(User, id=id)
    res.status = "reject"
    res.save()
    return redirect('hospital_staff_authorize')


def add_feedback(request):
    if request.method == "POST":
        message = request.POST.get('message')

        result = Feedback.objects.create(message=message, user=request.user)
        result.save()
        messages.success(request, "feedback successfully added.")
        return redirect('add_feedback')
    return render(request, 'add_feedback.html')


def feedback(request):
    user = Feedback.objects.all()
    return render(request, 'feedback.html', {'user':user})


def view_patient(request):
    result = Patient.objects.all()
    return render(request, 'view_patient.html', {'result': result})


def add_patient(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        gender = request.POST.get('gender')
        age = request.POST.get('age')
        
        result = Patient.objects.create(name=name, gender=gender, age=age, description=description)
        result.save()
        messages.success(request, "patient added successfully.")
        return redirect('view_patient')
    return render(request, 'add_patient.html')


def about(request):
    return render(request, 'about.html')



@csrf_exempt
def icd_ajax_predict(request):
    if request.method == "POST":
        try:
            # Lazy import to prevent heavy loading at module level
            from ml.predict import predict_icd
            import json

            data = json.loads(request.body)
            description = data.get("description", "")

            if not isinstance(description, str):
                return JsonResponse({"error": "Description must be a string"}, status=400)

            description = description.strip()
            if not description:
                return JsonResponse({"results": []})

            results = predict_icd(description)
            return JsonResponse({"results": results})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)



def edit_patient(request, id):
    result = get_object_or_404(Patient, id=id)
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        gender = request.POST.get('gender')
        age = request.POST.get('age')
        
        result.name = name
        result.gender = gender
        result.age = age
        result.description = description
        result.save()
        messages.success(request, "patient updated successfully.")
        return redirect('view_patient')
    return render(request, 'edit_patient.html', {'result': result})


def delete_patient(request, id):
    result = get_object_or_404(Patient, id=id)
    result.delete()
    messages.success(request, "patient deleted successfully.")
    return redirect('view_patient')