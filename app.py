import pandas as pd
import plotly.graph_objects as go
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import numpy as np

# Load Data
df = pd.read_csv('Datasets/electricity_appliance_wise_data.csv')  # Ensure this file exists
df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime
df.set_index('Date', inplace=True)  # Set Date as index

# Define Z-score function
def zscore(x, window=30):
    r = x.rolling(window=window)
    m = r.mean().shift(1)  # Rolling mean
    s = r.std(ddof=0).shift(1)  # Rolling standard deviation
    z = (x - m) / s  # Z-score calculation
    return z

# Calculate Z-scores for appliances
df['Fridge_zscore'] = zscore(df['Fridge'], 30)
df['AC_zscore'] = zscore(df['AC'], 30)
df['Kitchen_Appliances_zscore'] = zscore(df['Kitchen Appliances'], 30)
df['Washing_Machine_zscore'] = zscore(df['Washing Machine'], 30)
df['Other_Appliances_zscore'] = zscore(df['Other Appliances'], 30)

# Mark anomalies if Z-score > 3 (threshold can be adjusted)
df['Fridge_Anomaly'] = df['Fridge_zscore'] > 3
df['AC_Anomaly'] = df['AC_zscore'] > 3
df['Kitchen_Anomaly'] = df['Kitchen_Appliances_zscore'] > 3
df['Washing_Machine_Anomaly'] = df['Washing_Machine_zscore'] > 3
df['Other_Appliances_Anomaly'] = df['Other_Appliances_zscore'] > 3

# Initialize Dash App
app = dash.Dash(__name__)
app.title = 'Electricity Dashboard'

# App Layout
app.layout = html.Div([
    html.Div([
        html.H1("Watt-Wise", style={'textAlign': 'center', 'color': '#d3dce5'}),
        html.Img(src='/assets/logo.png', style={
            'position': 'absolute',
            'top': '10px',
            'right': '10px',
            'width': '100px',  # Adjust width as needed
            'height': 'auto'   # Maintain aspect ratio
        })
    ], style={'position': 'relative'}),  # This ensures the logo stays positioned
    
    dcc.Dropdown(
        id='view-selector',
        options=[
            {'label': 'Time-Series Plot', 'value': 'timeseries'},
            {'label': 'Appliance Anomalies', 'value': 'appliance_anomalies'}
        ],
        value='timeseries',
        style={'width': '50%', 'margin': 'auto', 'color': 'black'}
    ),
    dcc.Dropdown(
        id='appliance-selector',
        options=[
            {'label': 'Fridge', 'value': 'Fridge'},
            {'label': 'AC', 'value': 'AC'},
            {'label': 'Kitchen Appliances', 'value': 'Kitchen Appliances'},
            {'label': 'Washing Machine', 'value': 'Washing Machine'},
            {'label': 'Other Appliances', 'value': 'Other Appliances'}
        ],
        value='Fridge',
        style={'width': '50%', 'margin': 'auto', 'color': 'black', 'display': 'none'}  # Initially hidden
    ),
    dcc.Graph(id='graph')  # Placeholder for the graph
])

# Callback to Toggle Appliance Dropdown
@app.callback(
    Output('appliance-selector', 'style'),
    [Input('view-selector', 'value')]
)
def toggle_appliance_dropdown(selected_view):
    if selected_view == 'appliance_anomalies':
        return {'width': '50%', 'margin': 'auto', 'color': 'black'}  # Show dropdown
    else:
        return {'width': '50%', 'margin': 'auto', 'color': 'black', 'display': 'none'}  # Hide dropdown

# Callback to Update Graph
@app.callback(
    Output('graph', 'figure'),
    [Input('view-selector', 'value'),
     Input('appliance-selector', 'value')]
)
def update_graph(selected_view, selected_appliance):
    if selected_view == 'timeseries':
        # Time-Series Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Total_Consumption'],
                                 mode='lines', name='Actual Consumption',
                                 line_color="#19E2C5"))
        if 'Predicted_Consumption' not in df.columns:
            df['Predicted_Consumption'] = df['Total_Consumption'] * (0.9 + np.random.uniform(-0.05, 0.05, len(df)))  # Simulated predictions
        fig.add_trace(go.Scatter(x=df.index, y=df['Predicted_Consumption'],
                                 mode='lines', name='Predicted Consumption',
                                 line_color="#C6810B"))
        fig.update_layout(title="Time-Series Plot of Electricity Consumption",
                          xaxis_title="Date", yaxis_title="Consumption (kWh)",
                          template='plotly_dark', paper_bgcolor='rgba(0, 0, 0, 0)',
                          plot_bgcolor='rgba(0, 0, 0, 0)')
        return fig

    elif selected_view == 'appliance_anomalies' and selected_appliance:
        # Appliance-Specific Anomalies
        appliance_column = selected_appliance.replace(' ', '_')  # Match column naming convention
        anomaly_column = f"{appliance_column}_Anomaly"
        df_anomaly = df[df[anomaly_column]]

        if df_anomaly.empty:
            return go.Figure().update_layout(
                title=f"No anomalies detected for {selected_appliance}",
                xaxis_title="Date", yaxis_title="Consumption (kWh)",
                template='plotly_dark'
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[selected_appliance],
                                 mode='lines', name=f'{selected_appliance} Consumption',
                                 line_color="#19E2C5"))
        fig.add_trace(go.Scatter(x=df_anomaly.index, y=df_anomaly[selected_appliance],
                                 mode='markers', name='Anomalies',
                                 marker=dict(color='red', size=10)))  # Anomalies in red
        fig.update_layout(title=f"Anomalies in {selected_appliance} Consumption",
                          xaxis_title="Date", yaxis_title="Consumption (kWh)",
                          template='plotly_dark', paper_bgcolor='rgba(0, 0, 0, 0)',
                          plot_bgcolor='rgba(0, 0, 0, 0)')
        return fig

# Run App
if __name__ == '__main__':
    app.run(debug=True)
