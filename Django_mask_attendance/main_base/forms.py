from django.forms import ModelForm, fields
from django.contrib.auth.forms import UserCreationForm
from django import forms
from .models import UserProfile,Account



    

class createUserForm(UserCreationForm):
    email = forms.EmailField(max_length=50,help_text="Required. Add a valid email address")
    class Meta:
        model = Account
        fields = ['email','username','password1','password2']

class UserProfileForm(ModelForm):
    class Meta:
        model = UserProfile
        fields = ['date_of_birth','gender','department']