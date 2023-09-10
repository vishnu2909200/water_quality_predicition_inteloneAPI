# main web app


import requests
import streamlit as st
from streamlit_lottie import st_lottie
import numpy as np
import joblib
import pickle


#----------------------------------------------------------------------------------------------------------


intel=st.sidebar.selectbox('intel',["with oneAPI","without oneAPI"])
model_selection=st.sidebar.selectbox('model selection',["XGboost","random forest","logistic regression"])


#----------------------------------------------------------------------------------------------------------
# Load the trained model

#with 1 api
rf_1apimodel = pickle.load(open('randomForestmode_1apiskexdev.pk1','rb'))
xg_1apimodel = joblib.load(open('xgmodel_with1apidev.pk1','rb'))
log_1apimodel=joblib.load(open('logmodel_with1apidev.pk1','rb'))

#without 1 api
rfmodel = pickle.load(open('randomForestmode_without1api_job.pk1','rb'))
xgmodel = joblib.load(open('xgmodel_without1api_job.pk1','rb'))
logmodel=joblib.load(open('logmodel_without1api.pk1','rb'))

#-----------------------------------------------------------------------------------------------------------
st.title('WATER QUALITY PREDICITION')
#lottie
def load_lottieurl(url):
    r=requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

lottie_coding=load_lottieurl("https://lottie.host/20cf6858-72e1-4ff4-8ea0-3b1460dcf327/KJrS6NbRTB.json")

st_lottie(lottie_coding,height=300,key="coding")
#---------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------


# Your Streamlit app content

html_temp="""
<div style="background-color:lightblue;padding:1px">
<h2 style="color:white;text-align:center;">WATER QUALITY PREDICITION--intel ONEapi</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

#----------------------------------------------------------------------------------------------------------------------
# grt input from user

#catagorical value list
color_list=["Colorless","Faint Yellow","Light Yellow", "Near Colorless","Yellow"]
source_list=["Aquifer","Ground", "Lake","Reservoir","River","Spring","Stream","Well"]
month_list=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# Add user input elements  for input features

pH=st.number_input("pH",step=1.,format="%f")
Iron=st.number_input("Iron",step=1.,format="%f")
Nitrate=st.number_input("Nitrate",step=1.,format="%f")
Chloride=st.number_input("Chloride",step=1.,format="%f")
Lead=st.number_input("Lead",step=1.,format="%f")
Zinc=st.number_input("Zinc",step=1.,format="%f")

Color = st.selectbox("Select a Color",color_list)

Turbidity=st.number_input("Turbidity",step=1.,format="%f")
Fluoride=st.number_input("Fluoride",step=1.,format="%f")
Copper=st.number_input("Copper",step=1.,format="%f")
Odor=st.number_input("Odor",step=1.,format="%f")
Sulphate=st.number_input("Sulfate",step=1.,format="%f")
Conductivity=st.number_input("Conductivity",step=1.,format="%f")
Chlorine=st.number_input("Chlorine",step=1.,format="%f")
Manganese=st.number_input("Manganese",step=1.,format="%f")
DissolvedSolids=st.number_input("Total Dissolved Solids",step=1.,format="%f")

Source=st.selectbox("Select a Source",source_list)

WaterTemperature=st.number_input("Water Temperature",step=1.,format="%f")
AirTemperature=st.number_input("Air Temperature",step=1.,format="%f")

Month = st.selectbox("Select a Month",month_list)
Day=st.number_input("Day",step=1.,format="%f")

Time=st.number_input("Time of Day",step=1.,format="%f")

btn=st.button("PREDICT")

#--------------------------------------------------------------------------------------------------------------------
#convert user input to ml model input

catagoricaltonum_color=[0,0,0,0,0]
catagoricaltonum_source=[0,0,0,0,0,0,0,0]
catagoricaltonum_month=[0,0,0,0,0,0,0,0,0,0,0,0]

in_put=[pH,Iron,Nitrate,Chloride,Lead,Zinc,Turbidity,Fluoride,Copper,Odor,Sulphate,Conductivity,Chlorine,Manganese,DissolvedSolids,WaterTemperature,AirTemperature,Day,Time]

if Color:
    c=color_list.index(Color)
    catagoricaltonum_color[c]=1
if Source:
    s=source_list.index(Source)
    catagoricaltonum_source[s]=1
if Month:
    m=month_list.index(Month)
    catagoricaltonum_month[m]=1



#combin al list like numberical colums + catagorigal colum(color,source,month)

#1.color+source+month
catagoricaltonum_source.extend(catagoricaltonum_month)
catagoricaltonum_color.extend(catagoricaltonum_source)


in_put.extend(catagoricaltonum_color)

#---------------------------------------------------------------------------------------------------------------------
#get using intel1api or not
#get which ml model using
#then predict a result



if btn:
    pred=2
    if intel=="with oneAPI":
        if model_selection=="XGboost":
            pred=xg_1apimodel.predict(np.array([in_put]))
        elif model_selection=="random forest":
            pred=rf_1apimodel.predict(np.array([in_put]))
        elif model_selection=="logistic regression":
            pred=log_1apimodel.predict(np.array([in_put]))

    if intel=="without oneAPI":
        if model_selection=="XGboost":
            pred=xgmodel.predict(np.array([in_put]))
        elif model_selection=="random forest":
            pred=rfmodel.predict(np.array([in_put]))
        elif model_selection=="logistic regression":
            pred=logmodel.predict(np.array([in_put]))

     #result
            
    col1,col2,col3=st.columns(3)
    with col2:
        st.subheader('RESULT')
        if pred==1:
            html_temp="""
<div style="background-color:lightblue;padding:1px">
<h2 style="color:white;text-align:center;">PURE</h2>
</div>
"""
            st.markdown(html_temp, unsafe_allow_html=True)
            
        elif pred==0:
            html_temp="""
<div style="background-color:red;padding:1px">
<h2 style="color:white;text-align:center;">IMPURE</h2>
</div>
"""
            st.markdown(html_temp, unsafe_allow_html=True)
             
        else:
            st.success('some errors occure')       







#.................................................THANK YOU.......................................................





















#pred=model.predict(np.array([]))


#btn=st.button("PREDICT")
#ans=0
#pred=model.predict(np.array([[8.332988,0.000083,8.605777,122.799772,3.713298e-52,3.434827,0.022683,0.607283,0.144599,1.626212,87.266538,471.683357,3.708178,2.269945e-15,332.118789,19.129818,43.493324,29.0,4.0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0]]))
#ans=0 pred=model.predict(np.array([[8.091909,0.002167,9.925788,186.540872,4.171069e-132,3.807511,0.004867,0.222912,0.616574,0.795310,175.275175,385.025855,3.177849,3.296139e-03,168.075545,15.249416,69.336671,29.0,7.0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0]]))

#ans=1 pred=model.predict(np.array([[8.418457,0.000095,8.427576,256.570863,4.751543e-26,4.967504,3.824532,0.541850,0.284838,0.299860,371.261098,339.150786,2.630130,9.608234e-03,545.990529,21.436974,70.493951,8.0	,8.0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0]]))
#ans=1 pred=model.predict(np.array([[	8.273548	0.001236	10.182408	122.842038	4.226073e-33	0.903032	0.366909	3.076229	0.988735	3.268085	133.890373	718.942949	2.762962	2.824411e+00	112.458644	28.034083	18.158372	29.0 13.0		0	0	1	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	1	0	0	0	0]]))

#st.success(pred)

#if btn:
    #pred=model.predict(np.array([8.332988,0.000083,8.605777,122.799772,3.713298e-52,3.434827,0.022683,0.607283,0.144599,1.626212,87.266538,471.683357,3.708178,2.269945e-15,332.118789,19.129818,43.493324,29.0,4.0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0]))
    
    #st.subheader('RESULT')
    #if pred==1:
        #st.success('pure')
    #else:
        #st.success('impure')



# Make predictions using the model

#prediction = model.predict([[rainfall, temperature, humidity]])[0]

# Display the prediction to the user
#st.write(f'Predicted Water Level: {prediction} units')
#pH 0 Iron 0 Nitrate 0 Chloride 0 Lead 0 Zinc 0 Turbidity 0 Fluoride 0 Copper 0 Odor 0 Sulfate 0 Conductivity 0 Chlorine 0 Manganese 0 Total Dissolved Solids 0 Water Temperature 0 Air Temperature 0 Day 0 Time of Day 0 Target 0 Color_Colorless 0 Color_Faint Yellow 0 Color_Light Yellow 0 Color_Near Colorless 0 Color_Yellow 0 Source_Aquifer 0 Source_Ground 0 Source_Lake 0 Source_Reservoir 0 Source_River 0 Source_Spring 0 Source_Stream 0 Source_Well 0 Month_April 0 Month_August 0 Month_December 0 Month_February 0 Month_January 0 Month_July 0 Month_June 0 Month_March 0 Month_May 0 Month_November 0 Month_October 0 Month_September 0
#Color(1,0,0,0,0)
#source(0,0,0,0,0,0,1,0)
#month(0,0,0,0,1,0,0,0,0,0,0,0)
