#Librerias
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

#Cargar datos
data = pd.read_csv("data_clean.csv")
data = data.drop(columns=data.columns[1:4])
data = data.drop(columns=data.columns[2])




