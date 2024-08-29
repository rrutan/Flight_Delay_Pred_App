# %%
#%pip install streamlit

# %%
import streamlit as st
import joblib
import pandas as pd
import gdown

# %%
pipeline = joblib.load('xgboost_pipe.joblib')

# %%
url = f'https://drive.google.com/file/d/1n1n7RWlmNERfsby6zcKVNbnMEFl_W1w2/view?usp=sharing'
gdown.download(url, 'input2.csv', fuzzy=True, quiet=True)

df = pd.read_csv('input2.csv')
df = df.iloc[:,1:]
df.head()

features = df.columns.tolist()

numerical_cols = ['DISTANCE','ori_TMIN','ori_TMAX','ori_SNOW','ori_SNWD','ori_AWND','ori_PRCP','dest_TMIN','dest_TMAX','dest_SNOW','dest_SNWD',
    'dest_AWND','dest_PRCP','year','hour','day_of_week','month','day']
categorical_cols = ['AIRLINE','ORIGIN','DEST']

# %%
def app():
    st.markdown("<h3 style='text-align: center;'>Flight Delay Predictor powered by XGBoost</h3>", unsafe_allow_html=True)

    airline = st.selectbox('Airline', df['AIRLINE'].unique())
    origin = st.selectbox('Origin', df['ORIGIN'].unique())
    destination = st.selectbox('Destination',df['DEST'].unique())
    
    cond = (df['ORIGIN']== origin) & (df['DEST']== destination)
    dist_df = df[cond]

    if not dist_df.empty:
        distance = dist_df.iloc[0]['DISTANCE']
    else:
        st.write('The distance of your flight is not in our database, please google it and input distance.')
        distance = st.slider('Distance (miles)', 0,20000)
    
    #____________________________________________________________
    day_mapping = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7
    }

    def get_day_number(day_name):
        return day_mapping[day_name]

    days_of_week = list(day_mapping.keys())


    month_mapping = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
    }

    def get_month_number(month_name):
        return month_mapping[month_name]
    
    months = list(month_mapping.keys())
    #____________________________________________________________

    

    year = 2023
    selected_month_name = st.select_slider('Scheduled Month of Departure', options=months)
    month=get_month_number(selected_month_name)
    hour = st.slider('Scheduled Hour of Departure (0-23)', 0,23)
    day = st.slider('Scheduled Day of the Month of Departure (1-31)', 0,31)
    
    selected_day_name = st.select_slider('Scheduled Day of the Week of Departure', options=days_of_week)
    day_of_week=get_day_number(selected_day_name)
    
    ori_TMIN = ori_TMAX = ori_SNOW = ori_SNWD = ori_AWND = ori_PRCP = 0
    dest_TMIN = dest_TMAX = dest_SNOW = dest_SNWD = dest_AWND = dest_PRCP = 0


    cond2 = dist_df['month']==month

    w_df = dist_df[cond2]

    if not w_df.empty:

        ori_TMIN = w_df['ori_TMIN'].mean()
        ori_TMAX = w_df['ori_TMAX'].mean()
        ori_SNOW = w_df['ori_SNOW'].mean()
        ori_SNWD = w_df['ori_SNWD'].mean()
        ori_AWND = w_df['ori_AWND'].mean()
        ori_PRCP = w_df['ori_PRCP'].mean()

        dest_TMIN = w_df['dest_TMIN'].mean()
        dest_TMAX = w_df['dest_TMAX'].mean()
        dest_SNOW = w_df['dest_SNOW'].mean()
        dest_SNWD = w_df['dest_SNWD'].mean()
        dest_AWND = w_df['dest_AWND'].mean()
        dest_PRCP = w_df['dest_PRCP'].mean()

    else:
        st.write('No flight data in this month was found in our database')

    user_input = pd.DataFrame({
            'AIRLINE':[airline],
            'ORIGIN':[origin],
            'DEST':[destination],
            'DISTANCE': [distance],
            'ori_TMIN': [ori_TMIN],
            'ori_TMAX': [ori_TMAX],
            'ori_SNOW': [ori_SNOW],
            'ori_SNWD': [ori_SNWD],
            'ori_AWND': [ori_AWND],
            'ori_PRCP': [ori_PRCP],
            'dest_TMIN': [dest_TMIN],
            'dest_TMAX': [dest_TMAX],
            'dest_SNOW': [dest_SNOW],
            'dest_SNWD': [dest_SNWD],
            'dest_AWND': [dest_AWND],
            'dest_PRCP': [dest_PRCP],
            'year': [year],
            'month': [month],
            'day': [day],
            'day_of_week': [day_of_week],
            'hour': [hour]
        })

    pred = pipeline.predict(user_input)
    if pred[0] == 1:
        st.markdown("<p style='color:red; font-size:30px'><b>This flight is likely to be DELAYED</b></p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:green; font-size:30px'><b>This flight is likely to be ON TIME</b></p>", unsafe_allow_html=True)              
app()

    

# %%


# %%



