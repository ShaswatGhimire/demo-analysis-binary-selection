from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    pass
    id = models.BigAutoField(primary_key=True)

class Createnewlist(models.Model):
    id_value = models.IntegerField() 
    prediction_rf = models.BooleanField(default=False,blank=True,null=True)
    prediction_svm = models.BooleanField(default=False,blank=True,null=True)
    prediction_dt = models.BooleanField(default=False,blank=True,null=True)
    prediction_knn = models.BooleanField(default=False,blank=True,null=True)
    image = models.ImageField(null=True, blank = True)
    id = models.BigAutoField(primary_key=True)

