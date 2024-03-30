from django.contrib.auth import authenticate, login, logout
from django.db import IntegrityError
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.db import models
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage


import joblib
import pandas as pd
import numpy as np

import re

from PIL import Image 

from .models import User,Createnewlist


def index(request):
    items = Createnewlist.objects.all()
        
    return render(request, "auctions/index.html",{
        "items":items
    })


def login_view(request):
    if request.method == "POST":

        # Attempt to sign user in
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)

        # Check if authentication successful
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse("index"))
        else:
            return render(request, "auctions/login.html", {
                "message": "Invalid username and/or password."
            })
    else:
        return render(request, "auctions/login.html")


def logout_view(request):
    logout(request)
    return HttpResponseRedirect(reverse("index"))


def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]

        # Ensure password matches confirmation
        password = request.POST["password"]
        confirmation = request.POST["confirmation"]
        if password != confirmation:
            return render(request, "auctions/register.html", {
                "message": "Passwords must match."
            })

        # Attempt to create new user
        try:
            user = User.objects.create_user(username, email, password)
            user.save()
        except IntegrityError:
            return render(request, "auctions/register.html", {
                "message": "Username already taken."
            })
        login(request, user)
        return HttpResponseRedirect(reverse("index"))
    else:
        return render(request, "auctions/register.html")

def preprocessor(text):
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\b(?:the|and|is|it|of|in|to|for|with|on|at|by|this|an|a)\b', ' ', text, flags=re.IGNORECASE)
    return text

@login_required(login_url='/login/')
def suicidalprediction(request):
    if request.method == "POST":
        tweet = request.POST.get("tweet")

        suicidalknn = joblib.load('./mymodels/suicidal_knn.pkl')
        suicidalrf = joblib.load('./mymodels/suicidal_rf.pkl')
        suicidalsvm = joblib.load('./mymodels/suicidal_svm.pkl')
        suicidaldt = joblib.load('./mymodels/suicidal_dt.pkl')
        tfidf = joblib.load('./mymodels/suicidal_tfidf.joblib')

        mlmodels = {
        "Random Forest Classifier" : suicidalrf,
        "Decision Tree Classifier" : suicidaldt,
        "KNN Classifier" : suicidalknn,
        "SVM" : suicidalsvm
        }

        results = {}




        processed_text = preprocessor(tweet)

        #tfidf
        X_tfidf = tfidf.transform([processed_text]).toarray()

        for item in mlmodels:
            # Make predictions
            predicted_label = mlmodels[item].predict(X_tfidf)[0]
            print("model: ", item)
            print(f"Predicted Label: {predicted_label}")
            results[item] = predicted_label

        thing, created = Createnewlist.objects.get_or_create(id_value = 1)
        for item in results:
            print("result is: ", results[item])
            if item == "Random Forest Classifier":
                thing.prediction_rf = results[item]
            elif item == "Decision Tree Classifier":
                thing.prediction_dt = results[item]
            elif item == "KNN Classifier":
                thing.prediction_knn = results[item]
            elif item == "SVM":
                thing.prediction_svm = results[item]
        thing.save()
        
        item = Createnewlist.objects.get(id_value = 1)
        return render(request, "auctions/suicidalprediction.html", {"item":item})
    else:
        return render(request, "auctions/suicidalprediction.html")

@login_required(login_url='/login/')
def breastcancerdetection(request):
    if request.method == "POST":
        radiusmean = request.POST.get("radiusmean")
        texturemean = request.POST.get("texturemean")
        perimetermean = request.POST.get("perimetermean")
        areamean = request.POST.get("areamean")
        smoothnessmean = request.POST.get("smoothnessmean")
        compactnessmean = request.POST.get("compactnessmean")
        concavitymean = request.POST.get("concavitymean")
        concavepointsmean = request.POST.get("concavepointsmean")
        symmetrymean = request.POST.get("symmetrymean")
        radiusse = request.POST.get("radiusse")
        perimeterse = request.POST.get("perimeterse")
        arease = request.POST.get("arease")
        compactnessse = request.POST.get("compactnessse")
        concavityse = request.POST.get("concavityse")
        concavepointsse = request.POST.get("concavepointsse")
        radiusworst = request.POST.get("radiuswrost")
        textureworst = request.POST.get("texturewrost")
        perimeterworst = request.POST.get("perimeterwrost")
        areaworst = request.POST.get("areawrost")
        smoothnessworst = request.POST.get("smoothnesswrost")
        compactnessworst = request.POST.get("compactnesswrost")
        concavityworst = request.POST.get("concavitywrost")
        concavepointsworst = request.POST.get("concavepointswrost")
        symmetryworst = request.POST.get("symmetrywrost")
        fractualdimensionswrost = request.POST.get("fractualdimensionswrost")

        
        
        bcancerknn = joblib.load('./mymodels/bcancer_knn.pkl')
        bcancerrf = joblib.load('./mymodels/bcancer_rf.pkl')
        bcancersvm = joblib.load('./mymodels/bcancer_svm.pkl')
        bcancerdt = joblib.load('./mymodels/bcancer_dt.pkl')
        scaler = joblib.load('./mymodels/bcancer_scaler.joblib')

        mlmodels = {
        "Random Forest Classifier" : bcancerrf,
        "Decision Tree Classifier" : bcancerdt,
        "KNN Classifier" : bcancerknn,
        "SVM" : bcancersvm
        }

        results = {}
        data = [
            [radiusmean, texturemean, perimetermean, areamean, smoothnessmean, compactnessmean, concavitymean, concavepointsmean, symmetrymean, radiusse, perimeterse, arease, compactnessse, concavityse, concavepointsse, radiusworst, textureworst, perimeterworst, areaworst, smoothnessworst, compactnessworst, concavityworst, concavepointsworst, symmetryworst, fractualdimensionswrost]
        ]

        # Assuming column names are not provided, you may need to assign column names accordingly
        columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'radius_se', 'perimeter_se', 'area_se', 'compactness_se', 'concavity_se', 'concave points_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

        # Construct DataFrame   
        X = pd.DataFrame(data, columns=columns)
        X_scaled = scaler.transform(X)

        for item in mlmodels:
            # Make predictions
            predicted_label = mlmodels[item].predict(X_scaled)[0]
            print("model: ", item)
            print(f"Predicted Label: {predicted_label}")
            results[item] = predicted_label

        thing, created = Createnewlist.objects.get_or_create(id_value = 2)
        for item in results:
            print("result is: ", results[item])
            if item == "Random Forest Classifier":
                thing.prediction_rf = results[item]
            elif item == "Decision Tree Classifier":
                thing.prediction_dt = results[item]
            elif item == "KNN Classifier":
                thing.prediction_knn = results[item]
            elif item == "SVM":
                thing.prediction_svm = results[item]
        thing.save()
        
        item = Createnewlist.objects.get(id_value = 2)
        return render(request, "auctions/breastcancerdetection.html", {"item":item})
    else:
        return render(request, "auctions/breastcancerdetection.html")
    
@login_required(login_url='/login/')
def fraudulentjob(request):
    if request.method == "POST":
        has_location = 'has_location' in request.POST
        has_salary_range = 'has_salary_range' in request.POST
        has_description = 'has_description' in request.POST
        has_requirements = 'has_requirements' in request.POST
        has_benefits = 'has_benefits' in request.POST
        has_company_profile = 'has_company_profile' in request.POST
        has_functions = 'has_functions' in request.POST
        title = request.POST.get("title")
        function = request.POST.get("function")
        hascompanylogo = 'hascompanylogo' in request.POST
        cleanedtext = request.POST.get("cleanedtext")


      

        fraudulentjobknn = joblib.load('./mymodels/fraudulentjob_knn.pkl')
        fraudulentjobrf = joblib.load('./mymodels/fraudulentjob_rf.pkl')
        fraudulentjobsvm = joblib.load('./mymodels/fraudulentjob_svm.pkl')
        fraudulentjobdt = joblib.load('./mymodels/fraudulentjob_dt.pkl')

        mlmodels = {
        "Random Forest Classifier" : fraudulentjobrf,
        "Decision Tree Classifier" : fraudulentjobdt,
        "KNN Classifier" : fraudulentjobknn,
        "SVM" : fraudulentjobsvm
        }

        
        results = {}
        
        data = [has_location, has_salary_range, has_description, has_requirements,has_benefits, has_company_profile, has_functions, title, function, hascompanylogo, cleanedtext ]
        print(data)

        # Assuming column names are not provided, you may need to assign column names accordingly
        columns = ['location_isna', 'salary_range_isna', 'description_isna', 'requirements_isna', 'benefits_isna', 'company_profile_isna', 'function_isna', 'title', 'function', 'has_company_logo', 'cleaned_text']
        
        # Construct DataFrame   
        X = pd.DataFrame([data], columns=columns)
        print("i am here")
        print(X)


        

        for item in mlmodels:
            # Make predictions
            predicted_label = mlmodels[item].predict(X)[0]
            print("model: ", item)
            print(f"Predicted Label: {predicted_label}")
            results[item] = predicted_label

        thing, created = Createnewlist.objects.get_or_create(id_value = 3)
        for item in results:
            print("result is: ", results[item])
            if item == "Random Forest Classifier":
                thing.prediction_rf = results[item]
            elif item == "Decision Tree Classifier":
                thing.prediction_dt = results[item]
            elif item == "KNN Classifier":
                thing.prediction_knn = results[item]
            elif item == "SVM":
                thing.prediction_svm = results[item]
        thing.save()
        
        item = Createnewlist.objects.get(id_value = 3)
        return render(request, "auctions/fraudulentjob.html", {"item":item})
    else:
        return render(request, "auctions/fraudulentjob.html")
    


@login_required(login_url='/login/')
def dogsvscats(request):
    if request.method == "POST":
        print(request.FILES)
        img = request.FILES["sentFile"]
        file_name = "img.jpg"

        # Save the uploaded file locally
        with open(file_name, 'wb') as f:
            for chunk in img.chunks():
                f.write(chunk)

        # Open the saved file using PIL
        image = np.array(Image.open(file_name).convert('L'))
        image = np.resize(image, (32,32))
        image = image / 255.
        image = image.reshape((1, 32 * 32))

        suicidalknn = joblib.load('./mymodels/dogsvscats_knn.pkl')
        suicidalrf = joblib.load('./mymodels/dogsvscats_rf.pkl')
        suicidalsvm = joblib.load('./mymodels/dogsvscats_svm.pkl')
        suicidaldt = joblib.load('./mymodels/dogsvscats_dt.pkl')

        mlmodels = {
        "Random Forest Classifier" : suicidalrf,
        "Decision Tree Classifier" : suicidaldt,
        "KNN Classifier" : suicidalknn,
        "SVM" : suicidalsvm
        }

        results = {}

        for item in mlmodels:
            # Make predictions
            predicted_label = mlmodels[item].predict(image)[0]
            print("model: ", item)
            print(f"Predicted Label: {predicted_label}")
            results[item] = predicted_label

        thing, created = Createnewlist.objects.get_or_create(id_value = 4)
        for item in results:
            print("result is: ", results[item])
            if item == "Random Forest Classifier":
                thing.prediction_rf = results[item]
            elif item == "Decision Tree Classifier":
                thing.prediction_dt = results[item]
            elif item == "KNN Classifier":
                thing.prediction_knn = results[item]
            elif item == "SVM":
                thing.prediction_svm = results[item]
        thing.save()
        
        item = Createnewlist.objects.get(id_value = 4)
        return render(request, "auctions/dogsvscats.html", {"item":item})
    else:
        return render(request, "auctions/dogsvscats.html")