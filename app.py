import streamlit as st
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import google.generativeai as genai
import os

from sklearn.preprocessing import LabelEncoder


current_directory = os.path.dirname(__file__)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

kmeans_rf = joblib.load(os.path.join(current_directory, 'kmeans_rf_model.pkl'))
df = pd.read_csv(os.path.join(current_directory, 'german_credit_data.csv'))

label_encoders = {}
label_encoders['Sex'] = LabelEncoder()
df['Sex'] = label_encoders['Sex'].fit_transform(df['Sex'])

label_encoders['Job'] = LabelEncoder()
df['Job'] = label_encoders['Job'].fit_transform(df['Job'])

label_encoders['Housing'] = LabelEncoder()
df['Housing'] = label_encoders['Housing'].fit_transform(df['Housing'])

label_encoders['Purpose'] = LabelEncoder() 
df['Purpose'] = label_encoders['Purpose'].fit_transform(df['Purpose'])


st.set_page_config(page_title="German Credit Risk Prediction", page_icon=":guardsman:")
st.title("German Credit Risk Prediction with Clustering")

tabs = st.radio("Select Page", ['Exploratory Data Analysis', 'Risk Prediction'])

if tabs == 'Exploratory Data Analysis':
    st.header("Basic Information and Statistics")
    st.write(df.info())
    st.write(df.describe())

    st.subheader("Univariate Analysis")

    st.subheader("Age Distribution:")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Credit Amount Distribution:")
    fig, ax = plt.subplots()
    sns.histplot(df['Credit amount'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])  
    fig, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Custom Analysis: Average Credit Amount by Job")
    fig, ax = plt.subplots()
    df.groupby('Job')['Credit amount'].mean().plot(kind='bar', ax=ax)
    ax.set_ylabel("Average Credit Amount")
    ax.set_xlabel("Job")
    ax.set_title("Avg Credit Amount per Job Type")
    st.pyplot(fig)

    from PIL import Image
    import os

    current_directory = os.path.dirname(__file__)
    image = Image.open(os.path.join(current_directory, 'feature importance.png'))
    st.image(image, caption='Feature Importance', use_container_width=True)


elif tabs == 'Risk Prediction':
    st.header("Credit Risk Prediction App")


    age = st.number_input('Age', min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), step=1)
    
    sex = st.selectbox('Sex', options=['male', 'female'])
    
    
    job = st.selectbox('Job', options=df['Job'].unique())

    
    housing = st.selectbox('Housing', options=['own', 'free', 'rent'])

    
    saving_accounts = st.selectbox('Saving Accounts', options=['unknown', 'little', 'quite rich', 'rich', 'moderate'])
    
    
    checking_account = st.selectbox('Checking Account', options=['little', 'moderate', 'unknown', 'rich'])
    
    
    purpose = st.selectbox('Purpose', options=['radio/TV', 'education', 'furniture/equipment', 'car', 'business',
                                               'domestic appliances', 'repairs', 'vacation/others'])

    
    credit_amount = st.number_input('Credit Amount', min_value=0, step=100)
    duration = st.number_input('Duration (in months)', min_value=1, step=1)

    # Encode user inputs
    sex = label_encoders['Sex'].transform([sex])[0]
    job = label_encoders['Job'].transform([job])[0]
    housing = label_encoders['Housing'].transform([housing])[0]
    purpose = label_encoders['Purpose'].transform([purpose])[0]  

    
    label_encoders['Saving accounts'] = LabelEncoder()
    df['Saving accounts'] = label_encoders['Saving accounts'].fit_transform(df['Saving accounts'].fillna('unknown'))
    
    label_encoders['Checking account'] = LabelEncoder()
    df['Checking account'] = label_encoders['Checking account'].fit_transform(df['Checking account'].fillna('unknown'))

    saving_accounts = label_encoders['Saving accounts'].transform([saving_accounts])[0]
    checking_account = label_encoders['Checking account'].transform([checking_account])[0]

    
    user_input = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
        'Housing': [housing],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Saving accounts': [saving_accounts],
        'Checking account': [checking_account],
        'Purpose': [purpose]
    })

    if st.button('Predict Risk'):
        
        cluster = kmeans_rf.predict(user_input)[0]
        st.write(f"The applicant belongs to Cluster {cluster}.")

        
        if cluster == 1:
            st.write("The applicant might be a high-risk individual.")
            model = genai.GenerativeModel('gemini-1.5-flash')
            user_input_string = user_input.to_string(index=False)
            prompt=user_input_string+". and the applicant might be a high-risk individual. Now just give some personalized retention strategy for this applicant. keep it concise and impactful."
            response = model.generate_content(prompt)
            st.write(response.text)
        else:
            st.write("The applicant might be a low-risk individual.")
