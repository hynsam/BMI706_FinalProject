import altair as alt
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



@st.cache
def load_data():
    df = pd.read_csv("data_preprocessing/processed_data.csv")
    return df

df = load_data()

# a version of df where all varaibles are number, if categorical, they are encoded as numbers
df_num = df.copy()
for col in df_num.select_dtypes(include=['object']).columns:
    df_num[col] = df_num[col].astype('category').cat.codes

st.write("## Correlation between lifestyle-related features and health outcomes")

lifestyle_factors = ['Sleep Duration (hours)', 'Frequency of Muscle-Strengthening Activities per Week', 'Ever Had at Least 12 Alcoholic Drinks in One Year', 'Current Smoking Frequency', 'Time Spent Watching TV or Videos (minutes/day)', 'Calcium Intake (mg)', 'Alcohol Intake (grams)', 'Sugar Intake (grams)', 'Total Fat Intake (grams)', 'Engaged in Vigorous Activity in Past 30 Days', 'Frequency of Vigorous Physical Activity per Week', 'Smoked at Least 100 Cigarettes in Life']
health_outcomes = ['White Blood Cell Count (10^3 cells/uL)', 'Body Mass Index (BMI)', 'Now Taking Hypertension Medication', 'Ever Told Had Chlamydia', 'Lymphocyte Percentage (%)', 'Hemoglobin (g/dL)', 'Hematocrit (%)', 'Dentition Examination Status', 'Ever Told Had High Blood Pressure', 'Blood Mercury (µg/L)', 'Urine Creatinine (mg/dL)', 'Number of Prescription Medications Taken', 'Urinary Albumin (mg/L)', 'Diastolic Blood Pressure (mmHg)', 'Serum Creatinine (mg/dL)', 'Ever Told Had Genital Herpes', 'Albumin-Creatinine Ratio (mg/g)', 'Blood Lead (µg/dL)', 'Systolic Blood Pressure (mmHg)', 'Calculated LDL Cholesterol (mg/dL)', 'Weight (kg)', 'Triglycerides (mg/dL)', 'Mean Corpuscular Volume (fL)', 'Platelet Count (10^3 cells/uL)', 'Total Energy Intake (kcal)', 'Red Cell Distribution Width (%)', 'Red Blood Cell Count (million cells/uL)', 'Use Any Prescription Medications', 'Total Cholesterol (mg/dL)', 'Doctor Told You Have Diabetes', 'Time Since Last Dental Visit', 'Waist Circumference (cm)', 'Glycohemoglobin (A1C) %', 'Blood Selenium (µg/L)']
demographics = ['Gender of the Participant', 'Pregnancy Status at Examination', 'Family Poverty Income Ratio', 'Marital Status', 'Age of the Participant (years)', 'Full Sample 2-Year Interview Weight', 'Household Size', 'Country of Birth', 'Race/Ethnicity', 'Examination Month Period', 'Education Level (Adults 20+)', 'Full Sample 2-Year MEC Exam Weight']

health_outcomes_fig2 = st.multiselect(
    "Select Health Outcome",
    options=health_outcomes,
    default=['Use Any Prescription Medications', 'Weight (kg)', 'Dentition Examination Status', 'Doctor Told You Have Diabetes', 'Body Mass Index (BMI)'],
)

lifestyle_factor_fig2 = st.multiselect(
    "Select Lifestyle Factor",
    options=lifestyle_factors,
    default=['Frequency of Muscle-Strengthening Activities per Week', 'Frequency of Vigorous Physical Activity per Week', 'Calcium Intake (mg)', 'Alcohol Intake (grams)', 'Sugar Intake (grams)'],
)



# plot the correlation matrix as a heatmap
st.write("### Correlation Matrix")
# calculate the correlation matrix between the df_num[lifestyle_factor_fig2] and df_num[health_outcomes_fig2]
correlation_matrix = df_num[lifestyle_factor_fig2 + health_outcomes_fig2].corr()
# x-axis only show lifestyle factors, y-axis only show health outcomes
correlation_matrix = correlation_matrix.loc[health_outcomes_fig2, lifestyle_factor_fig2]

selector = alt.selection_point(on='click')

base = alt.Chart(correlation_matrix.reset_index().melt(id_vars='index')).add_params(
    selector
)

correlation_matrix_heatmap = base.mark_rect().encode(
    x=alt.X('variable:O', title='Lifestyle Factor'),
    y=alt.Y('index:O', title='Health Outcome'),
    # color=alt.Color('value:Q', title='Correlation')
    color=alt.condition(
        selector,
        alt.Color('value:Q', title='Correlation'),
        alt.value('lightgray')
    )
).properties(
    title='Correlation between Lifestyle Factors and Health Outcomes',
    width=600,
    height=600
)

# make a scatter plot of the selected data, 

chart = alt.vconcat(
    correlation_matrix_heatmap,
    selected_data
)

chart