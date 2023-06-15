from django.urls import path
from autoforex_ml.views import *

app_name = 'ml'

urlpatterns = [
    path('callData', callData, name='callData'),
    path('predictEUR', getPredictEUR, name='eur'),
    path('predictSGD', getPredictSGD, name='sgd'),
    path('predictUSD', getPredictUSD, name='usd'),
    path('predictJPY', getPredictJPY, name='JPY'),
    path('predictGBP', getPredictGBP, name='GBP'),
    path('predictAUD', getPredictAUD, name='AUD'),
    path('predictCAD', getPredictCAD, name='CAD'),
    path('predictCNY', getPredictCNY, name='CNY'),
    path('predictMYR', getPredictMYR, name='MYR'),
    path('predictRUB', getPredictRUB, name='RUB'),
]