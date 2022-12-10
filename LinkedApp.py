import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


st.markdown("# Predicting LinkedIn Users")
st.image("https://gujaratwebdesign.com/wp-content/uploads/2020/11/LinkedIn-Groups-.jpg",caption="https://gujaratwebdesign.com/linkedin-ads/",
        width=500)
st.markdown("#### Let's predict the liklihood a person is a LinkedIn User based on a few attributes")
st.header("Complete each User Attribute")

col1,col2=st.columns(2)

with col1:
##### EDUCATION
    educ = st.selectbox("Completed Education level", 
             options = ["Less than high school",
                        "High school incomplete",
                        "High school graduate",
                        "Some College, no degree",
                        "Two-year associate degree",
                        "Four-year degree",
                        "Some Postgraduate",
                        "Postgraduate Degree"])

#st.write(f"Education (pre-conversion): {educ}")

#st.write("**Convert Selection to Numeric Value**")

    if educ == "Less than high school":
        educ = 1
    elif educ == "High school incomplete":
        educ = 2
    elif educ == "High school graduate":
        educ = 3
    elif educ == "Some College, no degree":
        educ = 4
    elif educ == "Two-year associate degree":
        educ = 5
    elif educ == "Four-year degree":
        educ = 6
    elif educ == "Some Postgraduate":
        educ = 7
    elif educ == "Postgraduate Degree":
        educ = 8
    else:
        educ = 9
    
#st.write(f"Education (post-conversion): {educ}")

##### INCOME
    incom = st.selectbox("Income Range", 
             options = ["Less than $10,000",
                        "$10k to under $20,000",
                        "$20k to under $30,000",
                        "$30k to under $40,000",
                        "$40k to under $50,000",
                        "$50k to under $75,000",
                        "$75k to under $100,000",
                        "$100k to under $150,000",
                        "$150,000 or more?"])

#st.write(f"Income (pre-conversion): {incom}")

#st.write("**Convert Selection to Numeric Value**")

    if incom == "Less than $10,000":
        incom = 1
    elif incom == "$10k to under $20,000":
        incom = 2
    elif incom == "$20k to under $30,000":
        incom = 3
    elif incom == "$30k to under $40,000":
        incom = 4
    elif incom == "$40k to under $50,000":
        incom = 5
    elif incom == "$50k to under $75,000":
        incom = 6
    elif incom == "$75k to under $100,000":
        incom = 7
    elif incom == "$100k to under $150,000":
        incom = 8
    else:
        incom = 9
    
#st.write(f"Income (post-conversion): {incom}")

##### PARENT
    par = st.selectbox("Parent?", 
             options = ["No",
                        "Yes"])

#st.write(f"Parent (pre-conversion): {par}")

#st.write("**Convert Selection to Numeric Value**")

    if par == "Yes":
        par = 1
    else:
        par = 2
    
#st.write(f"Parent (post-conversion): {par}")

with col2:
##### MARRIED
    marital = st.selectbox("Married?", 
             options = ["No",
                        "Yes"])

#st.write(f"Married (pre-conversion): {marital}")

#st.write("**Convert Selection to Numeric Value**")

    if marital == "Yes":
        marital = 1
    else:
        marital = 2
    
#st.write(f"Married (post-conversion): {marital}")

 ##### Gender
    gender = st.selectbox("Gender", 
             options = ["Female",
                        "Male"])

#st.write(f"Gender(pre-conversion): {gender}")

#st.write("**Convert Selection to Numeric Value**")

    if gender == "Female":
        gender = 1
    else:
        gender = 2

#st.write(f"Gender (post-conversion): {gender}")

##### Age
    age = st.number_input("Age",
            min_value=1,
            max_value=99,
            value=25)

#st.write("How old are you?: ", age)

#### DATA and PREDICTIONS

s = pd.read_csv(
    "C:\\Users\\aarik\\Desktop\\Prog II\\social_media_usage.csv")

def clean_sm(x):
    return(np.where(x == 1,
            1,
            0))


ss = pd.DataFrame({
    "income":np.where(s["income"] > 9, np.nan,s["income"]),
    "education":np.where(s["educ2"] > 8,np.nan,s["educ2"]),
    "parent":np.where(s["par"] == 1,1,0),
    "married":np.where(s["marital"] == 1,1,0),
    "female":np.where(s["gender"] == 2,1,0),
    "age":np.where(s["age"] > 98,np.nan,s["age"]),
    "sm_li": clean_sm(s["web1h"])
})

ss = ss.dropna()

y = ss["sm_li"]
x = ss[["income","education","parent","married","female","age"]]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,  
                                                    test_size=0.2,    
                                                    random_state=159) 


lr = LogisticRegression(class_weight='balanced')

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)


new_person = pd.DataFrame({
    "income": [incom],
    "education":[educ],
    "parent":[par],
    "married":[marital],
    "female":[gender],
    "age":[age]
})

probability= lr.predict_proba(new_person)[0][1]

new_person["predict_linkedin"] = lr.predict(new_person)

probability=round(probability*100,1)

if probability >= 75:
    then = "Very likely"
elif probability >= 50:
    then = "Likely"
else:
    then = "Not likely"


#st.write

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = probability,
    title = {'text': f"{then} a LinkedIn User"},
    gauge = {"axis": {"range": [0,100]},
            "steps": [
                {"range": [0,35], "color":"navy"},
                {"range": [36, 64], "color":"gray"},
                {"range": [65, 100], "color":"lightblue"}
            ],
            "bar":{"color":"lightgray"}}
))


st.plotly_chart(fig)



