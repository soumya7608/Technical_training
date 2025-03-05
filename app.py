import pandas as pd
import time # To delay response

import warnings 
warnings.filterwarnings("ignore") # To supress warnings

import streamlit as st  # UI Module

import joblib # Saved Models Load Module
import pickle # Saved Encodings Load Module

model = joblib.load("jobsvmreg.pkl")
scaler = joblib.load("sc.pkl")

# Loading Saved Ordinal Encoded files
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
    
data = pd.read_csv("ml_clean_data.csv")

st.title("Predicted Minimum Fee and maximum fee")
st.image("https://www.indiafilings.com/learn/wp-content/uploads/2019/07/Skill-Training-for-Rural-Youth.jpg")

st.subheader("Input data train on ....")
st.dataframe(data.head())

st.selectbox("select institute name :",data["Institute name"].unique())
st.selectbox("select Area :",data["Area"].unique())
st.selectbox("select City :",data["City"].unique())
st.selectbox("select Aws and devops :",data["Aws and devops"].unique())
st.selectbox("select Java full stack :",data["Java full stack"].unique())
st.selectbox("select Data science and AI :",data["Data science and AI"].unique())
st.selectbox("select python :",data["python"].unique())
st.selectbox("select Faculty Expertise :",data["Faculty Expertise"].unique())
st.selectbox("select Training Methodology :",data["Training Methodology"].unique())
st.selectbox("select Placement Rate :",data["Placement Rate"].unique())
st.number_input(f"select Average Salary (in lpa) range :{data['Average Salary in lpa'].min()} to {data['Average Salary in lpa'].max()}:")
st.selectbox("select Placement Support :",data["Placement Support"].unique())
st.selectbox("select Internship Opportunities :",data["Internship Opportunities"].unique())
st.selectbox("select Certifications Offered :",data["Certifications Offered"].unique())
st.number_input(f"Enter review (out of 5): {data['review out of 5'].min()} to {data['review out of 5'].max()}:")


st.subheader("Final predicted Data....")

if st.button("Maximum fee and muinimum fee:"):
    inpdata = pd.read_csv("CampaInpData.csv")
      
    user_inputs = {}  # Dictionary to store user responses
    
    for col in inpdata.columns:
        unique_values = inpdata[col].unique()  # Get unique values for the column

        if inpdata[col].dtype in ['int64', 'float64']:
            user_inputs[col] = st.number_input(
                f"Enter {col} (Range: {inpdata[col].min()} to {inpdata[col].max()})",
                min_value=float(inpdata[col].min()), 
                max_value=float(inpdata[col].max())
            )
        else:
            user_inputs[col] = st.selectbox(f"Select {col}:", unique_values) 
            
    # Convert user inputs to a DataFrame
    row = pd.DataFrame([user_inputs])
    
    
    # Feature Engineering: Need to apply same steps done for training, while giving it to model for prediction

    binary_cols = ["Aws and devops", "Java full stack", "Data science and AI", "python","Certifications Offered","Internship Opportunities"]
    row[binary_cols] = row[binary_cols].apply(lambda x: x.map({"yes": 1, "no": 0}))

    row.City.replace({'hyderabad':0,'banglore':1}, inplace=True)  
    row['Training Methodology'].replace({'online and  offline training':0, 'offline training':1,'online training':2},inplace=True)
    row["Placement Rate"].replace({"average":0, "good":1}, inplace=True)
    row["Faculty Expertise"].replace({"experience":1, "well qualified":0}, inplace=True)    
    def categorize_support(text):
      text = str(text).lower()  # Convert to lowercase for uniformity
      
      categories = {
          'Resume_Assistance': any(keyword in text for keyword in ['resume', 'linkedin']),
          'Interview_Training': any(keyword in text for keyword in ['interview', 'soft skills']),
          'Job_Placement': any(keyword in text for keyword in ['job placement', 'placement assistance']),
          'Project_Training': any(keyword in text for keyword in ['real time projects', 'internship', 'project'])
      }
    
      return pd.Series(categories)

     # Apply the grouping function
    df_encoded = row['Placement Support'].apply(categorize_support)
    row = pd.concat([row, df_encoded], axis=1)
    row = row.drop(columns=['Placement Support'])
    job = ['Resume_Assistance','Interview_Training', 'Job_Placement', 'Project_Training']
    row[job] = row[job].astype(bool)
    row[job] = row[job].apply(lambda x: x.map({True: 1, False: 0}))

    # LabelEncoding
    for col in ["Institute name", "Area"]:
      #row[f"{col}"] = label_encoder.transform(row[col])
        # Transform using the existing encoder
      row[f"{col}"] = label_encoder.fit_transform(row[col])


    # Scaling
    numerical_cols = ["Institute name", "Area","Training Methodology", "Average Salary in lpa", "review out of 5"]

    # Apply scaling only to numerical columns
    row[numerical_cols] = scaler.transform(row[numerical_cols])
    
    # Prediction
     # **Prediction**
    min_fee, max_fee = model.predict(row)[0]

    # **Display Results**
    st.write(f"\n🎯 Predicted Min Fee: {round(min_fee, 2)} LPA")
    st.write(f"🎯 Predicted Max Fee: {round(max_fee, 2)} LPA")