import streamlit as st
from streamlit_shap import st_shap
import shap
import joblib
import xgboost
import numpy as np
from scipy import special

model_path = 'Model\XGBRegressor_model.joblib'
model = joblib.load(model_path)
lambda_value = 1.3613740501599898

st.title('Blueberry Crop Yield Prediction')

with st.form('Prediction Form'):
    st.header('Set the bee, weather and fruit parameters:')
    clonesize = st.slider('Size of clone: ',min_value = 0.3,max_value = 0.7, step = 0.01, format = '%f')
    bumbles = st.slider('Density of bumblebees: ',min_value = 0.0,max_value = 0.6, step = 0.01, format = '%f')
    andrena = st.slider('Density of andrena bees: ',min_value = 0.0,max_value = 0.8, step = 0.01, format = '%f')
    osmia = st.slider('Density of osmia bees: ',min_value = 0.0,max_value = 0.8, step = 0.01, format = '%f')
    MaxOfUpperTRange = st.slider('Upper Temperature Limit: ',min_value = 86.0,max_value = 95.0, step = 0.01, format = '%f')
    AverageRainingDays = st.slider('Average of raining days: ',min_value = 0.0,max_value = 0.6, step = 0.01, format = '%f')
    fruitset = st.slider('Set of fruits: ',min_value = 0.1, max_value = 0.7, step = 0.01, format = '%f')
    
    submit_values = st.form_submit_button ('Predict')
    
if submit_values:
    data = np.array([clonesize, bumbles,andrena, osmia, MaxOfUpperTRange,AverageRainingDays, fruitset]).reshape(1,-1)
    feature_names = [clonesize, bumbles,andrena, osmia, MaxOfUpperTRange,AverageRainingDays, fruitset]
    prediction = model.predict(data)
    prediction = special.inv_boxcox(prediction, lambda_value)
    
    st.header('Here is the predicted yield')
    st.success(f'{prediction[0]}')
    st.balloons()
    
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    st_shap(shap.plots.beeswarm(shap_values, order=shap_values.abs.max(0)))
    st_shap(shap.plots.waterfall(shap_values[0])