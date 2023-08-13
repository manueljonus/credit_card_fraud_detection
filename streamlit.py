import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Read data
df=pd.read_csv(r"C:\Users\manuel jonus\Desktop\Luminar Technolab\Luminar data sets\credit_dataset.csv")
df1=df.drop(["Unnamed: 0","ID","FLAG_MOBIL","NO_OF_CHILD"],axis=1)

# Data preprocessing
from sklearn.preprocessing import LabelEncoder
le_1= LabelEncoder()
le_2= LabelEncoder()
le_3= LabelEncoder()
le_4= LabelEncoder()
le_5= LabelEncoder()
le_6= LabelEncoder()
le_7= LabelEncoder()

df["GENDER"] = le_1.fit_transform(df["GENDER"])
df["CAR"] = le_2.fit_transform(df["CAR"])
df["REALITY"] = le_3.fit_transform(df["REALITY"])
df["INCOME_TYPE"] = le_4.fit_transform(df["INCOME_TYPE"])

df["EDUCATION_TYPE"] = le_5.fit_transform(df["EDUCATION_TYPE"])
df["FAMILY_TYPE"] = le_6.fit_transform(df["FAMILY_TYPE"])
df["HOUSE_TYPE"] = le_7.fit_transform(df["HOUSE_TYPE"])



# le=LabelEncoder()
# lst = ["GENDER", "CAR", "REALITY", "INCOME_TYPE", "EDUCATION_TYPE", "FAMILY_TYPE", "HOUSE_TYPE"]
# for i in lst:
#     df[i] = le.fit_transform(df[i])

df.drop(["Unnamed: 0", "ID", "FLAG_MOBIL", "NO_OF_CHILD"], axis=1, inplace=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

os = SMOTE(random_state=1)
X_os, y_os = os.fit_resample(X, y)

mms = MinMaxScaler()
X_os = mms.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_os, y, test_size=0.3, random_state=1)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Streamlit application
import streamlit as st

st.markdown('<h1 style="color: #25A2F8;">Credit Card Fraud Detection</h1>', unsafe_allow_html=True)

nav = st.sidebar.radio("Navigation", ["Home", "Data", "Plotting Graph", "Prediction"])

if nav == "Home":
    st.image("https://i.postimg.cc/y6j10ZVt/i-Stock-1203763961.jpg", caption='Image Caption', use_column_width=True)

if nav == "Data":
    st.markdown('<h3 style="color: #011410;">Data Set</h3>', unsafe_allow_html=True)

    if st.checkbox("Show Data Set"):
        st.dataframe(df1, width=5000, height=1000)

    st.markdown('<h3 style="color: #011410;">Checking whether the data is balanced or imbalanced</h3>', unsafe_allow_html=True)
    if st.checkbox("Check the Data is balanced"):
        st.dataframe(df1["TARGET"].value_counts())

if nav == "Plotting Graph":
    import matplotlib.pyplot as plt

    labels = ["Not Fraud", "Fraud"]
    sizes = df["TARGET"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    st.pyplot(fig)

if nav == "Prediction":
    st.markdown('<h3 style="color: #011410;">Check Whether The Transaction Is Fraudulent Or Not.</h3>', unsafe_allow_html=True)

    st.title("Prediction App")

    input1 = st.selectbox('GENDER', ['F', 'M'])
    input2 =st.selectbox('CAR', ['Y', 'N'])
    input3 = st.selectbox('REALITY', ['Y', 'N'])
    input4 = st.number_input("INCOME")
    input5 = st.selectbox('INCOME_TYPE', ['Working', 'Commercial associate',"State servant","Pensioner","Student"])
    input6 = st.selectbox('EDUCATION_TYPE', ['Secondary / secondary special', 'Higher education',"Incomplete higher","Lower secondary","Academic degree"])
    input7 = st.selectbox('FAMILY_TYPE', ['Married', 'Single / not married',"Civil marriage","Separated","Widow"])
    input8 = st.selectbox('HOUSE_TYPE', ['House / apartment', 'With parents',"Municipal apartment","Rented apartment","Office apartment","Co-op apartment"])
    input9 = st.number_input("WORK_PHONE")
    input10 = st.number_input("PHONE")
    input11 = st.number_input("E_MAIL")
    input12 = st.number_input("FAMILY SIZE")
    input13 = st.number_input("BEGIN_MONTH")
    input14 = st.number_input("AGE")
    input15 = st.number_input("YEARS_EMPLOYED")

    if st.button("Predict"):
        input_data = pd.DataFrame({
            "Input 1": le_1.transform([input1]),
            "Input 2": le_2.transform([input2]),
            "Input 3": le_3.transform([input3]),
            "Input 4": [input4],
            "Input 5": le_4.transform([input5]),
            "Input 6": le_5.transform([input6]),
            "Input 7": le_6.transform([input7]),
            "Input 8": le_7.transform([input8]),
            "Input 9": [input9],
            "Input 10": [input10],
            "Input 11": [input11],
            "Input 12": [input12],
            "Input 13": [input13],
            "Input 14": [input14],
            "Input 15": [input15]})
        # input_data=[[1,1,1,112500.0,4,4,1,1,0,0,0,2.0,29,59,3]]
        input_data_scaled = mms.transform(input_data)  # Scale the input data

        prediction = rf.predict(input_data_scaled)
        # Reverse the prediction logic
        if prediction == 0:
            result = "Not Fraud"
            color = "green"
        else:
            result = "Fraud"
            color = "red"
        st.markdown(f'Prediction: <span style="color:{color}; font-size: 24px; display:inline;">{result}</span>', unsafe_allow_html=True)

        # st.write(result)