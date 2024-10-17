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

df["Obesity"] = (df['Body Mass Index (BMI)'] >= 30).astype(int)
df["Hypertension"] = ((df['Systolic Blood Pressure (mmHg)'] >= 130) | (df['Diastolic Blood Pressure (mmHg)'] >= 80) | 
                          (df['Ever Told Had High Blood Pressure'] == 'Yes') | (df['Now Taking Hypertension Medication'] == 'Yes')).astype(int)
df["Diabetes"] = ((df['Glycohemoglobin (A1C) %'] >= 6.5) | (df['Doctor Told You Have Diabetes'] == 'Yes')).astype(int)
df['High Cholesterol'] = ((df['Total Cholesterol (mg/dL)'] >= 240) | (df['Calculated LDL Cholesterol (mg/dL)'] >= 160) | 
                          (df['Triglycerides (mg/dL)'] >= 200) | (df['Now Taking Hypertension Medication'] == 'Yes')).astype(int)

# Title of the Plot
st.write("# Health Visualization Dashboard")

### Task 1 ###
st.write("## What is the Distribution of Disease across Demographics Groups?")

# Disease selector
diseases_list = [
    "Obesity",
    "Diabetes",
    "Hypertension",
    "High Cholesterol"
]
disease_option  = st.selectbox("Select a Disease", diseases_list)

# Age range slider
age_min, age_max  = st.slider("Select Age Range of Participants", 
                              min_value=int(df['Age of the Participant (years)'].min()), 
                              max_value=int(df["Age of the Participant (years)"].max()), 
                              value=(int(df['Age of the Participant (years)'].min()), int(df['Age of the Participant (years)'].max())))
# Filter dataset based on the selected age range
subset = df[(df['Age of the Participant (years)'] >= age_min) & (df['Age of the Participant (years)'] <= age_max)]

# a donut chat for the gender groups
gender_disease_counts = subset.groupby('Gender of the Participant')[[disease_option]].sum().reset_index()
gender_donut = alt.Chart(gender_disease_counts).mark_arc(innerRadius=50, outerRadius=90).encode(
    theta=alt.Theta(field=disease_option, type='quantitative', aggregate='sum'),
    color=alt.Color('Gender of the Participant:N', legend=alt.Legend(title="Gender"), 
                    scale=alt.Scale(domain=['Female', 'Male'], range=['#4C78A8', '#F28E2C'])),
    tooltip=[alt.Tooltip('Gender of the Participant:N', title='Gender'),
             alt.Tooltip(field=disease_option, title=f'{disease_option} Cases')]
).properties(
    title=f'Number of {disease_option} Cases by Gender',
    width=250
)
# a donut chat for the race groups
race_disease_counts = subset.groupby('Race/Ethnicity')[[disease_option]].sum().reset_index()
race_donut = alt.Chart(race_disease_counts).mark_arc(innerRadius=50, outerRadius=90).encode(
    theta=alt.Theta(field=disease_option, type='quantitative', aggregate='sum'),
    color=alt.Color('Race/Ethnicity:N', legend=alt.Legend(title="Race")),
    tooltip=[alt.Tooltip('Race/Ethnicity:N', title='Race'),
             alt.Tooltip(field=disease_option, title=f'{disease_option} Cases')]
).properties(
    title=f'Number of {disease_option} Cases by Race',
    width=250
)

donut = alt.hconcat(gender_donut, race_donut).resolve_scale(
    # two donut charts should use different color schema
    color='independent'
)

age_disease_counts = subset.groupby('Age of the Participant (years)')[[disease_option]].sum().reset_index()
age_distribution_chart = alt.Chart(age_disease_counts).mark_bar().encode(
    x=alt.X('Age of the Participant (years):O', title="Age of Participants", axis=alt.Axis(labelAngle=0)),
    y=alt.Y(disease_option + ':Q', title=f'Number of {disease_option} Cases'),
    tooltip=[alt.Tooltip('Age of the Participant (years):O', title='Age'),
             alt.Tooltip(field=disease_option, title=f'{disease_option} Cases')]
).properties(
    title=f'Distribution of {disease_option} Cases by Age',
    width=600
)

# Calculate the prevalence as percentages within each group
grouped_df = subset.groupby(["Gender of the Participant", "Race/Ethnicity"]).agg(
    total = ('Respondent', 'size'),
    disease_total = (disease_option, 'sum')
).reset_index()
grouped_df['prevalence'] = (grouped_df['disease_total'] / grouped_df['total']) * 100

all_genders = subset.groupby("Race/Ethnicity").agg(
    total=('Respondent', 'size'),
    disease_total=(disease_option, 'sum')
).reset_index()
all_genders['Gender of the Participant'] = 'All'
all_genders['prevalence'] = (all_genders['disease_total'] / all_genders['total']) * 100
final_df = pd.concat([grouped_df, all_genders])

# Create a selection component
race_selection = alt.selection_multi(fields=['Race/Ethnicity'], bind='legend')

# Create the prevalence chart with bars
prevalence_chart = alt.Chart(final_df).mark_bar().encode(
    x=alt.X('Race/Ethnicity:N', title=None, axis=alt.Axis(labels=False)),
    y=alt.Y('prevalence:Q', title='Prevalence (%)'),
    color=alt.condition(race_selection, 'Race/Ethnicity:N', alt.value('lightgray')),
    tooltip=[alt.Tooltip('Race/Ethnicity:N', title='Race/Ethnicity'),
             alt.Tooltip('prevalence:Q', title='Prevalence (%)', format='.2f')]
).add_selection(
    race_selection
).properties(width = 200)

# Add text labels showing the prevalence on top of each bar
text = prevalence_chart.mark_text(
    align='center',
    baseline='bottom',
    dy=-3  # Adjust the vertical position of the text
).encode(
    text=alt.Text('prevalence:Q', format='.2f')  # Format to 2 decimal places
)

# Layer the bars and text labels together
layered_prevalence_chart = alt.layer(prevalence_chart, text)

# Now facet the layered chart by gender
faceted_chart = layered_prevalence_chart.facet(
    column=alt.Column('Gender of the Participant:N', header=alt.Header(title="Gender", labelOrient='bottom'))
).resolve_scale(
    x='independent'
).properties(
    title=f'Prevalence of {disease_option} by Sex and Race'
)

# Display the chart in Streamlit
st.altair_chart(donut)
st.altair_chart(age_distribution_chart)
st.altair_chart(faceted_chart)

### Task 1 ###

# a version of df where all varaibles are number, if categorical, they are encoded as numbers
df_num = df.copy()
for col in df_num.select_dtypes(include=['object']).columns:
    df_num[col] = df_num[col].astype('category').cat.codes

st.write("## What is the Correlation between each Lifestyle Factors and Health Outcome?")

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

correlation_matrix_heatmap


### task 3
st.write("## Does Individual with Similar Lifestyle Factors have Similar Health Outcomes?")

lifestyle_factors_of_interest = st.multiselect(
    "Select Lifestyle Factors",
    options=lifestyle_factors,
    default=['Sugar Intake (grams)', 'Total Fat Intake (grams)', 'Alcohol Intake (grams)', 'Current Smoking Frequency', 'Smoked at Least 100 Cigarettes in Life']
)
health_outcome_of_interest = st.selectbox(
    "Select Health Outcome",
    options=health_outcomes,
    index=health_outcomes.index('Weight (kg)')
)

# pca on all the lifestyle factors
X = df_num[lifestyle_factors_of_interest]
scaler = StandardScaler()
X = X.dropna()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Health Outcome'] = df[health_outcome_of_interest]

# drop all nan
df_pca = df_pca.dropna()

# Brush for selection
brush = alt.selection_interval()

# if health_outcome_of_interest is continuous, Health Outcome:Q, else Health Outcome:O
if df[health_outcome_of_interest].dtype == 'object':
    # Scatter Plot
    points = alt.Chart(df_pca).mark_circle().encode(
        x='PC1:Q',
        y='PC2:Q',
        color=alt.condition(brush, alt.Color('Health Outcome:N', legend=alt.Legend(title=f'{health_outcome_of_interest}')), alt.value('grey')),
        tooltip=['Health Outcome', 'PC1', 'PC2']
    ).properties(
        title=f'PCA Plot of Lifestyle Factors for \"{health_outcome_of_interest}\"',
        width=600,
        height=600
    ).add_params(
        brush
    )
    # for each category, calculate the percentage of each category in the selected
    ranked_text = alt.Chart(df_pca).mark_text(align='left', fontSize=16).encode(
            y=alt.Y('row_number:O').axis(None)
        ).transform_filter(
            brush
        )
    categories = df_pca['Health Outcome'].unique()
    text_lst = []
    for category in categories:
        
        ranked_text.encode(text='mean(Health Outcome)').properties(
            title=alt.Title(text='Mean Health Outcome', align='left')
        )
        
        temp_text = ranked_text.transform_filter(
            alt.datum['Health Outcome'] == category
            ).encode(
                text='count(Health Outcome)'
            ).properties(
                title=alt.Title(text=f'{category} Count', align='left')
            )
        text_lst.append(temp_text)
        
    # histogram for the selected data
    health_outcome_hist = alt.Chart(df_pca).mark_bar().encode(
        x=alt.X('Health Outcome:N', title=f'{health_outcome_of_interest}'),
        y='count()',
        color=alt.value('steelblue')
    ).transform_filter(
        brush
    ).properties(
        title=f'Distribution of {health_outcome_of_interest} in Selected Data',
        width=600,
        height=200
    )
    text = alt.vconcat(*text_lst)
    points & text & health_outcome_hist
else:
    # Scatter Plot
    points = alt.Chart(df_pca).mark_circle().encode(
        x='PC1:Q',
        y='PC2:Q',
        color=alt.condition(brush, alt.Color('Health Outcome:Q', legend=alt.Legend(title=f'{health_outcome_of_interest}')), alt.value('grey')),
        tooltip=['Health Outcome', 'PC1', 'PC2']
    ).properties(
        title=f'PCA Plot of Lifestyle Factors for \"{health_outcome_of_interest}\"',
        width=600,
        height=600
    ).add_params(
        brush
    )
    ranked_text = alt.Chart(df_pca).mark_text(align='left', fontSize=16).encode(
            y=alt.Y('row_number:O').axis(None)
        ).transform_filter(
            brush
        )
    mean_Health_Outcome = ranked_text.encode(text='mean(Health Outcome)').properties(
            title=alt.Title(text=f'Mean {health_outcome_of_interest}', align='left')
        )
    median_Health_Outcome = ranked_text.encode(text='median(Health Outcome)').properties(
            title=alt.Title(text=f'Median {health_outcome_of_interest}', align='left')
        )
    std_Health_Outcome = ranked_text.encode(text='stdev(Health Outcome)').properties(            
            title=alt.Title(text=f'Standard Deviation {health_outcome_of_interest}', align='left')
        )
    text = alt.vconcat(
            mean_Health_Outcome,
            median_Health_Outcome,
            std_Health_Outcome,
        )
    # add a histogram for the selected data
    health_outcome_hist = alt.Chart(df_pca).mark_bar().encode(
        x=alt.X('Health Outcome:Q', bin=alt.Bin(maxbins=30), title=f'{health_outcome_of_interest}'),
        y='count()',
        color=alt.value('steelblue')
    ).transform_filter(
        brush
    ).properties(
        title=f'Distribution of {health_outcome_of_interest} in Selected Data',
        width=600,
        height=200
    )
    # align point and text to stack vertically
    points & text & health_outcome_hist
    


