import streamlit as st
import joblib
import pandas as pd

st.write("# Machine Learning based Breast Cancer Prediction")


# streamlit run streamlit_fhs.py


#Create input boxes:

col1, col2 = st.columns(2)

texture_mean = col1.number_input("Texture mean",step=1.,format="%.5f")
area_mean = col2.number_input("Area mean",step=1.,format="%.5f")

smoothness_mean = col1.number_input("Smoothness mean",step=1.,format="%.5f")
concavity_mean = col2.number_input("Concavity mean",step=1.,format="%.5f")

symmetry_mean = col1.number_input("Symmetry mean",step=1.,format="%.5f")
texture_se = col2.number_input("Texture standard error",step=1.,format="%.5f")

area_se = col1.number_input("Area standard error",step=1.,format="%.5f")
smoothness_se = col2.number_input("Smoothness standard error",step=1.,format="%.5f")

concavity_se = col1.number_input("Concavity standard error",step=1.,format="%.5f")
symmetry_se = col2.number_input("Symmetry standard error",step=1.,format="%.5f")

fractal_dimension_se =col1.number_input("Fractal Dimension standard error",step=1.,format="%.5f")
smoothness_worst = col2.number_input("Smmothness worst",step=1.,format="%.5f")

symmetry_worst = col1.number_input("Symmetry worst",step=1.,format="%.5f")
fractal_dimension_worst = col2.number_input("Fractal dimension worst",step=1.,format="%.5f")

# st.button('Predict')

df_predict = pd.DataFrame([[texture_mean,area_mean,smoothness_mean,concavity_mean,symmetry_mean,texture_se,area_se,smoothness_se,concavity_se,symmetry_se,
                fractal_dimension_se,smoothness_worst,symmetry_worst,fractal_dimension_worst]])

print(df_predict)

columns = ["texture_mean","area_mean","smoothness_mean","concavity_mean","symmetry_mean","texture_se","area_se","smmothness_se","concavity_se","symmetry_se",
                          "fractal_dimension_se","smoothness_worst", "symmetry_worst","fractal_dimension_worst"]
print(columns)

model = joblib.load('/Users/sivagamiaravind/breastcancer/rf-model.pkl')
prediction = model.predict(df_predict)

if st.button('Predict'):

    if(prediction[0]==0):
        st.write('<p class="big-font">Predicted Class: Benign.</p>',unsafe_allow_html=True)

    else:
        st.write('<p class="big-font">Predicted Class: Malignant</p>',unsafe_allow_html=True)
 