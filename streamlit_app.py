import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import sklearn as skl
from sklearn import linear_model
from scipy.special import expit

# Import LabelEncoder
from sklearn import preprocessing

# Import Naive Bayes models
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
import re

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
# from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

# from GetForest import *
import joblib
from math import floor

contact_column = "Company Industry"

##### Data Cleaning #######
# Fill Empty Management Levels
def fill_management(df, management_str="Custom Field 3\n(Management Level)"):
    for index, row in df.iterrows():
        if (
            row[management_str] != "C-Level"
            and row[management_str] != "VP-Level"
            and row[management_str] != "Director"
            and row[management_str] != "Manager"
            and row[management_str] != "Non-Manager"
        ):
            # hard-coded right now, should do some regex
            # if "CEO" or "Co-Founder" in str(row['Title']):
            if "CEO" in str(row["Title"]):  # no instances
                df[management_str].loc[index] = "C-Level"
            elif "Vice President" in str(row["Title"]):
                df[management_str].loc[index] = "VP-Level"
            elif "Director" in str(row["Title"]):  # no instances
                df[management_str].loc[index] = "Director"
            elif "Manager" in str(row["Title"]):
                df[management_str].loc[index] = "Manager"
            else:
                df[management_str].loc[index] = "Non-Manager"
    return df


###### Import Model ######

rf_load = joblib.load("savedFiles/rf.joblib")
# Import DataFrame with correct column names for model
empty_df = joblib.load("savedFiles/empty_df_listOfColumnNames.joblib")

######################
# Start Streamlit code

st.title("TactiCALLS: Prospect Fit Predictor")


@st.cache(suppress_st_warning=True)
def load(csv_filename):
    try:
        with open(csv_filename) as input:
            st.text(input.read())
    except FileNotFoundError:
        st.error("File not found.")

    return pd.read_csv(csv_filename)
    # should make a try


st.write("Import CSV with prospect info")

# take user input csv file
potential_client_csv_file = st.text_input("Enter file path:", "demo.csv")
# change this to be whatever csv format for exports out of Zoominfo (or whatever Dylan uses)

# contact_info = pd.read_csv(potential_client_csv_file)
contact_info = load(potential_client_csv_file)
df_contacts = pd.DataFrame(contact_info)
test_final_set = df_contacts

categorical_features_list = [
    "Company Industry",
    "Title",
    "Job Function",
    "Management Level",
]
continuous_features_list = ["Company Size", "Revenue"]

raw_df = contact_info[categorical_features_list + continuous_features_list]


test_dummies = pd.get_dummies(
    test_final_set[categorical_features_list], prefix=categorical_features_list
)
test_dummies = pd.concat(
    [test_dummies, test_final_set[["Company Size"]], test_final_set[["Revenue"]]], 1
)

final_test_dataframe = empty_df.append(test_dummies).fillna(0)
final_test_dataframe = final_test_dataframe[empty_df.columns]
# st.dataframe(final_test_dataframe)

predicted = rf_load.predict(final_test_dataframe)
prob = rf_load.predict_proba(final_test_dataframe)
prob_df = pd.DataFrame(prob, columns=["No Quote Prob.", "Quote Request Prob."])
prob_df = pd.concat(
    [test_final_set[[contact_column]], prob_df, test_final_set[["Quote Requested"]]], 1
)

sorted_prob = prob_df.sort_values(by=["Quote Request Prob."], ascending=False)

# st.write('Predicted Quote Request: ',predicted,' with probabilies: ', prob)# '\n size encoded: ',listOfFeatureEncoders[0].transform(company_size_input) )


sorted_prob = sorted_prob.reset_index()
prediction_df = sorted_prob[["Quote Request Prob.", contact_column]]

# User input export filename
export_csv_file = st.text_input("Enter export file name:", "ProspectPredictions.csv")
# User interactive export button
export = st.button("Export")
if export:
    sorted_prob.to_csv(export_csv_file)
    st.write("Exported to: ", export_csv_file)
    st.dataframe(prediction_df.head(10))
# Display model output sorted by prediction
if st.checkbox("Show full sorted predictions", key="pred"):
    st.dataframe(prediction_df)

# Display input CSV
if st.checkbox("Show raw data", key="raw"):
    st.dataframe(raw_df)

# Calculate conversion rates for comparisons/ impact analysis
total_rows = sorted_prob.shape[0]

average_conversion = sorted_prob["Quote Requested"].mean()

frac = 0.1
top_x = floor(frac * total_rows)
top_conversion = sorted_prob.iloc[0:top_x].mean()[-1]

min_convert_prob = 0.5
thres_rate_df = sorted_prob.loc[sorted_prob["Quote Request Prob."] >= min_convert_prob]
thres_rate_conversion = thres_rate_df.mean()[-1]

# st.write("Average Conversion Rate:", average_conversion)
# st.write("Top ",frac*100, "% Conversion Rate:", top_conversion)
# st.write(thres_rate_df.shape[0], " out of ", total_rows ," possible contacts with Probability over ",min_convert_prob*100 )
# st.write("There conversion rate would have been:", thres_rate_conversion)


# End Streamlit code
######################
