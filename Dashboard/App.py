#%% 
#Librerias
import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

import airbnb_modelos as m

data = m.data

# %%

# Matriz de diseño y listas de variables
X = m.X
numeric_vars = m.numeric_vars
categorical_vars = m.categorical_vars
all_vars = m.all_vars   # variables numericas + categoricas

# Modelos
reg_model = m.model                  # regresión lineal
mejor_modelo_reg = m.mejor_modelo    # NN regresión
scaler_reg = m.scaler                # StandardScaler de la NN de regresión

mejor_modelo_clf = m.mejor_modelo_clf   # NN clasificación
scaler_clf = m.scaler_clf               # StandardScaler de la NN de clasificación
umbral_precio = m.umbral_precio         # umbral de precio para recomendar

#%%

#Filtros para el dash

if "neighbourhood_cleansed" in data.columns:
    neighbourhood_options = sorted(data["neighbourhood_cleansed"].dropna().unique())
else:
    neighbourhood_options = []

if "room_type" in data.columns:
    room_type_options = sorted(data["room_type"].dropna().unique())
else:
    room_type_options = []

price_min = float(data["price"].min())
price_max = float(data["price"].max())
reviews_min = int(data["number_of_reviews"].min())
reviews_max = int(data["number_of_reviews"].max())

def filtrar_df(neighbourhoods, room_types, price_range, reviews_range):
    dff = data.copy()

    if price_range is None:
        price_range = [price_min, price_max]
    if reviews_range is None:
        reviews_range = [reviews_min, reviews_max]

    if neighbourhoods and "neighbourhood_cleansed" in dff.columns:
        dff = dff[dff["neighbourhood_cleansed"].isin(neighbourhoods)]

    if room_types and "room_type" in dff.columns:
        dff = dff[dff["room_type"].isin(room_types)]

    dff = dff.dropna(subset=["price", "number_of_reviews"])

    dff = dff[
        (dff["price"] >= price_range[0])
        & (dff["price"] <= price_range[1])
        & (dff["number_of_reviews"] >= reviews_range[0])
        & (dff["number_of_reviews"] <= reviews_range[1])
    ]

    return dff


# %%

# DISEÑO DE LA APP

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    style={"backgroundColor": "#f2f2f2", "fontFamily": "Arial, sans-serif"},
    children=[
        # ENCABEZADO
        html.Div(
            style={
                "backgroundColor": "#2c3e50",
                "padding": "15px",
                "color": "white",
                "textAlign": "center",
            },
            children=[
                html.H1("Dashboard Airbnb CDMX – modelos Uniandes"),
                html.P(
                    "Explora el mercado y usa los tres modelos: regresión lineal, red neuronal regresión y red neuronal clasificación."
                ),
            ],
        ),

        # FILTROS
        html.Div(
            style={"padding": "20px"},
            children=[
                html.Div(
                    style={"display": "flex", "gap": "20px"},
                    children=[
                        html.Div(
                            style={"flex": "1"},
                            children=[
                                html.Label("Vecindarios:", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="filter-neighbourhood",
                                    options=[
                                        {"label": n, "value": n}
                                        for n in neighbourhood_options
                                    ],
                                    multi=True,
                                    placeholder="Seleccione vecindarios",
                                ),
                            ],
                        ),
                        html.Div(
                            style={"flex": "1"},
                            children=[
                                html.Label(
                                    "Tipos de habitación:", style={"fontWeight": "bold"}
                                ),
                                dcc.Dropdown(
                                    id="filter-room-type",
                                    options=[
                                        {"label": r, "value": r}
                                        for r in room_type_options
                                    ],
                                    multi=True,
                                    placeholder="Seleccione tipos de habitación",
                                ),
                            ],
                        ),
                    ],
                ),
                html.Br(),
                html.Div(
                    style={"display": "flex", "gap": "40px"},
                    children=[
                        html.Div(
                            style={"flex": "1"},
                            children=[
                                html.Label(
                                    "Rango de precio por noche (MXN):",
                                    style={"fontWeight": "bold"},
                                ),
                                dcc.RangeSlider(
                                    id="filter-price-range",
                                    min=price_min,
                                    max=price_max,
                                    step=50,
                                    value=[price_min, price_max],
                                ),
                            ],
                        ),
                        html.Div(
                            style={"flex": "1"},
                            children=[
                                html.Label(
                                    "Número de reviews:", style={"fontWeight": "bold"}
                                ),
                                dcc.RangeSlider(
                                    id="filter-reviews-range",
                                    min=reviews_min,
                                    max=reviews_max,
                                    step=1,
                                    value=[reviews_min, reviews_max],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # TABS
        dcc.Tabs(
            id="tabs-main",
            value="tab-1",
            children=[
                # TAB 1: DESCRIPTIVO
                dcc.Tab(
                    label="Análisis descriptivo",
                    value="tab-1",
                    children=[
                        html.Div(
                            style={"padding": "20px"},
                            children=[
                                html.Div(
                                    id="kpi-cards",
                                    style={
                                        "display": "flex",
                                        "justifyContent": "space-around",
                                        "marginBottom": "20px",
                                    },
                                ),
                                html.Div(
                                    style={"display": "flex"},
                                    children=[
                                        dcc.Graph(
                                            id="histogram-precio",
                                            style={"width": "50%"},
                                        ),
                                        dcc.Graph(
                                            id="box-roomtype-precio",
                                            style={"width": "50%"},
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                # TAB 2: FACTORES
                dcc.Tab(
                    label="Análisis de factores",
                    value="tab-2",
                    children=[
                        html.Div(
                            style={"padding": "20px", "display": "flex"},
                            children=[
                                dcc.Graph(
                                    id="heatmap-correlacion-precio",
                                    style={"width": "50%"},
                                ),
                                html.Div(
                                    style={"width": "50%", "paddingLeft": "20px"},
                                    children=[
                                        html.Label(
                                            "Analizar precio medio por:",
                                            style={"fontWeight": "bold"},
                                        ),
                                        dcc.Dropdown(
                                            id="dropdown-categoria-barra",
                                            options=[
                                                {
                                                    "label": "Vecindario",
                                                    "value": "neighbourhood_cleansed",
                                                },
                                                {
                                                    "label": "Tipo de habitación",
                                                    "value": "room_type",
                                                },
                                                {
                                                    "label": "Tipo de propiedad",
                                                    "value": "property_type",
                                                },
                                            ],
                                            value="neighbourhood_cleansed",
                                        ),
                                        dcc.Graph(
                                            id="bar-precio-categoria",
                                            style={"height": "500px"},
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                # TAB 3: SIMULADOR CON LOS 3 MODELOS
                dcc.Tab(
                    label="Simulador predictivo (3 modelos)",
                    value="tab-3",
                    children=[
                        html.Div(
                            style={"padding": "20px", "display": "flex"},
                            children=[
                                # PANEL INPUTS
                                html.Div(
                                    style={
                                        "width": "40%",
                                        "padding": "20px",
                                        "backgroundColor": "white",
                                        "borderRadius": "5px",
                                    },
                                    children=[
                                        html.H4("Características de la propiedad"),

                                        html.Label("Vecindario:"),
                                        dcc.Dropdown(
                                            id="sim-neighbourhood",
                                            options=[
                                                {"label": n, "value": n}
                                                for n in neighbourhood_options
                                            ],
                                            placeholder="neighbourhood_cleansed",
                                        ),
                                        html.Br(),

                                        html.Label("Tipo de propiedad:"),
                                        dcc.Dropdown(
                                            id="sim-property-type",
                                            options=[
                                                {"label": v, "value": v}
                                                for v in sorted(
                                                    data["property_type"]
                                                    .dropna()
                                                    .unique()
                                                )
                                            ]
                                            if "property_type" in data.columns
                                            else [],
                                            placeholder="property_type",
                                        ),
                                        html.Br(),

                                        html.Label("Tipo de habitación:"),
                                        dcc.Dropdown(
                                            id="sim-room-type",
                                            options=[
                                                {"label": v, "value": v}
                                                for v in room_type_options
                                            ],
                                            placeholder="room_type",
                                        ),
                                        html.Br(),

                                        html.Label("Número de huéspedes (accommodates):"),
                                        dcc.Input(
                                            id="sim-accommodates",
                                            type="number",
                                            value=2,
                                            min=1,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                        html.Br(), html.Br(),

                                        html.Label("Recámaras (bedrooms):"),
                                        dcc.Input(
                                            id="sim-bedrooms",
                                            type="number",
                                            value=1,
                                            min=0,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                        html.Br(), html.Br(),

                                        html.Label("Camas (beds):"),
                                        dcc.Input(
                                            id="sim-beds",
                                            type="number",
                                            value=1,
                                            min=0,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                        html.Br(), html.Br(),

                                        html.Label("Baños (bathrooms):"),
                                        dcc.Input(
                                            id="sim-bathrooms",
                                            type="number",
                                            value=1,
                                            min=0,
                                            step=0.5,
                                            style={"width": "100%"},
                                        ),
                                        html.Br(), html.Br(),

                                        html.Label("Noches mínimas (minimum_nights):"),
                                        dcc.Input(
                                            id="sim-min-nights",
                                            type="number",
                                            value=1,
                                            min=1,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                        html.Br(), html.Br(),

                                        html.Label("Número de reviews:"),
                                        dcc.Input(
                                            id="sim-num-reviews",
                                            type="number",
                                            value=10,
                                            min=0,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                        html.Br(), html.Br(),

                                        html.Label("Calificación promedio (review_scores_rating):"),
                                        dcc.Input(
                                            id="sim-rating",
                                            type="number",
                                            value=95,
                                            min=0,
                                            max=100,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                        html.Br(), html.Br(),

                                        html.Label("Disponibilidad 365 días (availability_365):"),
                                        dcc.Input(
                                            id="sim-availability",
                                            type="number",
                                            value=180,
                                            min=0,
                                            max=365,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                        html.Br(), html.Br(),

                                        html.Button(
                                            "Predecir con los 3 modelos",
                                            id="boton-predecir",
                                            n_clicks=0,
                                            style={
                                                "marginTop": "20px",
                                                "width": "100%",
                                                "backgroundColor": "#2c3e50",
                                                "color": "white",
                                                "padding": "10px",
                                            },
                                        ),
                                    ],
                                ),

                                # PANEL RESULTADOS
                                html.Div(
                                    style={"width": "60%", "paddingLeft": "40px"},
                                    children=[
                                        html.Div(
                                            id="output-prediccion",
                                            style={
                                                "fontSize": "20px",
                                                "fontWeight": "bold",
                                                "textAlign": "left",
                                                "padding": "20px",
                                                "backgroundColor": "white",
                                                "borderRadius": "5px",
                                                "whiteSpace": "pre-line",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ],
)

#endregion

#region CALLBACKS

# TAB 1: descriptivo
@app.callback(
    [
        Output("kpi-cards", "children"),
        Output("histogram-precio", "figure"),
        Output("box-roomtype-precio", "figure"),
    ],
    [
        Input("filter-neighbourhood", "value"),
        Input("filter-room-type", "value"),
        Input("filter-price-range", "value"),
        Input("filter-reviews-range", "value"),
    ],
)
def update_tab1(neigh_value, room_value, price_range, reviews_range):
    dff = filtrar_df(neigh_value, room_value, price_range, reviews_range)

    if dff.empty:
        fig_empty = go.Figure().update_layout(title="No hay datos para la selección.")
        return [], fig_empty, fig_empty

    avg_price = dff["price"].mean()
    total_listings = len(dff)
    avg_rating = dff["review_scores_rating"].mean()

    kpi_cards = [
        html.Div(
            style={"textAlign": "center","padding": "10px","backgroundColor": "white","borderRadius": "5px","width": "30%"},
            children=[html.H3(f"${avg_price:,.0f}"), html.P("Precio promedio")],
        ),
        html.Div(
            style={"textAlign": "center","padding": "10px","backgroundColor": "white","borderRadius": "5px","width": "30%"},
            children=[html.H3(f"{total_listings:,}"), html.P("Número de anuncios")],
        ),
        html.Div(
            style={"textAlign": "center","padding": "10px","backgroundColor": "white","borderRadius": "5px","width": "30%"},
            children=[html.H3(f"{avg_rating:.2f}"), html.P("Calificación promedio")],
        ),
    ]

    fig_hist = px.histogram(
        dff,
        x="price",
        nbins=50,
        title="Distribución del precio por noche",
        labels={"price": "Precio por noche (MXN)"},
    )
    fig_hist.add_vline(x=dff["price"].mean(),   line_dash="dash", line_color="red",   annotation_text="Media")
    fig_hist.add_vline(x=dff["price"].median(), line_dash="dot",  line_color="green", annotation_text="Mediana")

    fig_box = px.box(
        dff,
        x="room_type",
        y="price",
        title="Precio por tipo de habitación",
        labels={"room_type": "Tipo de habitación", "price": "Precio (MXN)"},
    )

    return kpi_cards, fig_hist, fig_box


# TAB 2: factores
@app.callback(
    [
        Output("heatmap-correlacion-precio", "figure"),
        Output("bar-precio-categoria", "figure"),
    ],
    [
        Input("filter-neighbourhood", "value"),
        Input("filter-room-type", "value"),
        Input("filter-price-range", "value"),
        Input("filter-reviews-range", "value"),
        Input("dropdown-categoria-barra", "value"),
    ],
)
def update_tab2(neigh_value, room_value, price_range, reviews_range, dropdown_value):
    dff = filtrar_df(neigh_value, room_value, price_range, reviews_range)

    if dff.empty:
        fig_empty = go.Figure().update_layout(title="No hay datos para la selección.")
        return fig_empty, fig_empty

    corr_cols = [c for c in numeric_vars if c in dff.columns] + ["price"]
    corr_matrix = dff[corr_cols].corr()
    corr_with_price = corr_matrix[["price"]].sort_values(by="price", ascending=False)
    corr_with_price = corr_with_price.drop("price", axis=0)

    fig_heat = px.imshow(
        corr_with_price,
        text_auto=".2f",
        title="Correlación de variables numéricas con el precio",
        labels=dict(color="Correlación"),
        aspect="auto",
    )
    fig_heat.update_traces(textfont_size=14)
    fig_heat.update_xaxes(showticklabels=False)

    if dropdown_value and dropdown_value in dff.columns and dff[dropdown_value].notna().any():
        bar_data = dff.groupby(dropdown_value)["price"].mean().reset_index()
        bar_data = bar_data.sort_values(by="price", ascending=False).head(20)
        fig_bar = px.bar(
            bar_data,
            x="price",
            y=dropdown_value,
            orientation="h",
            title=f"Top 20 – Precio promedio por {dropdown_value.replace('_', ' ').title()}",
            labels={"price": "Precio promedio (MXN)",
                    dropdown_value: dropdown_value.replace("_", " ").title()},
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fig_bar = go.Figure().update_layout(title="No hay datos para la selección.")

    return fig_heat, fig_bar


# TAB 3: simulador con tus 3 modelos
@app.callback(
    Output("output-prediccion", "children"),
    [Input("boton-predecir", "n_clicks")],
    [
        State("sim-neighbourhood", "value"),
        State("sim-property-type", "value"),
        State("sim-room-type", "value"),
        State("sim-accommodates", "value"),
        State("sim-bedrooms", "value"),
        State("sim-beds", "value"),
        State("sim-bathrooms", "value"),
        State("sim-min-nights", "value"),
        State("sim-num-reviews", "value"),
        State("sim-rating", "value"),
        State("sim-availability", "value"),
    ],
    prevent_initial_call=True,
)
def simular_modelos(n_clicks,
                    neigh, prop_type, room_type,
                    accommodates, bedrooms, beds, bathrooms,
                    min_nights, num_reviews, rating, availability):

    if n_clicks is None:
        raise PreventUpdate

    # Defaults
    if accommodates is None: accommodates = 2
    if bedrooms    is None: bedrooms    = 1
    if beds        is None: beds        = bedrooms
    if bathrooms   is None: bathrooms   = 1
    if min_nights  is None: min_nights  = 1
    if num_reviews is None: num_reviews = 10
    if rating      is None: rating      = 95
    if availability is None: availability = 180

    # Fila con las mismas variables que all_vars
    fila = {}

    # numéricas
    if "accommodates" in numeric_vars:          fila["accommodates"] = float(accommodates)
    if "bedrooms" in numeric_vars:             fila["bedrooms"] = float(bedrooms)
    if "beds" in numeric_vars:                 fila["beds"] = float(beds)
    if "bathrooms" in numeric_vars:            fila["bathrooms"] = float(bathrooms)
    if "minimum_nights" in numeric_vars:       fila["minimum_nights"] = float(min_nights)
    if "availability_365" in numeric_vars:     fila["availability_365"] = float(availability)
    if "number_of_reviews" in numeric_vars:    fila["number_of_reviews"] = float(num_reviews)
    if "review_scores_rating" in numeric_vars: fila["review_scores_rating"] = float(rating)

    # categóricas
    if "property_type" in categorical_vars:
        fila["property_type"] = prop_type if prop_type is not None else data["property_type"].mode()[0]
    if "room_type" in categorical_vars:
        fila["room_type"] = room_type if room_type is not None else data["room_type"].mode()[0]
    if "neighbourhood_cleansed" in categorical_vars:
        fila["neighbourhood_cleansed"] = neigh if neigh is not None else data["neighbourhood_cleansed"].mode()[0]

    # asegurar todas las columnas de all_vars
    for c in all_vars:
        if c not in fila:
            if c in numeric_vars:
                fila[c] = float(data[c].median())
            else:
                fila[c] = data[c].mode()[0]

    df_sim_raw = pd.DataFrame([fila], columns=all_vars)

    # One-hot igual que X
    X_sim = pd.get_dummies(df_sim_raw, columns=categorical_vars, drop_first=True)
    X_sim = X_sim.reindex(columns=X.columns, fill_value=0)

    # 1) Regresión lineal
    price_lin = float(reg_model.predict(X_sim)[0])

    # 2) NN regresión
    X_sim_scaled_reg = scaler_reg.transform(X_sim)
    price_nn = float(mejor_modelo_reg.predict(X_sim_scaled_reg)[0][0])

    # 3) NN clasificación
    X_sim_scaled_clf = scaler_clf.transform(X_sim)
    proba_reco = float(mejor_modelo_clf.predict(X_sim_scaled_clf)[0][0])
    proba_reco = max(0.0, min(1.0, proba_reco))
    label_reco = "Recomendado" if proba_reco >= 0.5 else "No recomendado"

    texto = (
        f"Resultado de los 3 modelos:\n\n"
        f"- Regresión lineal (OLS): ${price_lin:,.0f} MXN\n"
        f"- Red neuronal (regresión): ${price_nn:,.0f} MXN\n"
        f"- Red neuronal (clasificación):\n"
        f"    · Probabilidad de ser RECOMENDADO: {proba_reco:.2%}\n"
        f"    · Etiqueta: {label_reco}\n"
        f"(Umbral de precio para recomendar: {umbral_precio:,.0f} MXN)"
    )

    return texto

#endregion

if __name__ == "__main__":
    app.run(debug=True)

# %%
