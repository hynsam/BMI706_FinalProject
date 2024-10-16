import altair as alt
import pandas as pd
import streamlit as st

### P1.2 ###

# Move this code into `load_data` function {{
# cancer_df = pd.read_csv("https://raw.githubusercontent.com/hms-dbmi/bmi706-2022/main/cancer_data/cancer_ICD10.csv").melt(  # type: ignore
#     id_vars=["Country", "Year", "Cancer", "Sex"],
#     var_name="Age",
#     value_name="Deaths",
# )

# pop_df = pd.read_csv("https://raw.githubusercontent.com/hms-dbmi/bmi706-2022/main/cancer_data/population.csv").melt(  # type: ignore
#     id_vars=["Country", "Year", "Sex"],
#     var_name="Age",
#     value_name="Pop",
# )

# df = pd.merge(left=cancer_df, right=pop_df, how="left")
# df["Pop"] = df.groupby(["Country", "Sex", "Age"])["Pop"].fillna(method="bfill")
# df.dropna(inplace=True)

# df = df.groupby(["Country", "Year", "Cancer", "Age", "Sex"]).sum().reset_index()
# df["Rate"] = df["Deaths"] / df["Pop"] * 100_000

# }}


@st.cache
def load_data():
    ## {{ CODE HERE }} ##
    cancer_df = pd.read_csv("https://raw.githubusercontent.com/hms-dbmi/bmi706-2022/main/cancer_data/cancer_ICD10.csv").melt(  
    id_vars=["Country", "Year", "Cancer", "Sex"],
    var_name="Age",
    value_name="Deaths",
    )

    pop_df = pd.read_csv("https://raw.githubusercontent.com/hms-dbmi/bmi706-2022/main/cancer_data/population.csv").melt( 
        id_vars=["Country", "Year", "Sex"],
        var_name="Age",
        value_name="Pop",
    )

    df = pd.merge(left=cancer_df, right=pop_df, how="left")
    df["Pop"] = df.groupby(["Country", "Sex", "Age"])["Pop"].fillna(method="bfill")
    df.dropna(inplace=True)

    df = df.groupby(["Country", "Year", "Cancer", "Age", "Sex"]).sum().reset_index()
    df["Rate"] = df["Deaths"] / df["Pop"] * 100_000
    return df


# Uncomment the next line when finished
df = load_data()

### P1.2 ###


st.write("## Age-specific cancer mortality rates")

### P2.1 ###
# replace with st.slider
year = st.slider(
    "Select Year",
    min_value=int(df["Year"].min()),
    max_value=int(df["Year"].max()),
    value=2012,  
)
subset = df[df["Year"] == year]
### P2.1 ###


### P2.2 ###
# replace with st.radio
sex = st.radio(
    "Select Sex",
    options=["M", "F"],
    index=0 
)
subset = subset[subset["Sex"] == sex]
### P2.2 ###


### P2.3 ###
# replace with st.multiselect
# (hint: can use current hard-coded values below as as `default` for selector)
countries = st.multiselect(
    "Select Countries",
    options=df["Country"].unique().tolist(), 
    default=[
        "Austria",
        "Germany",
        "Iceland",
        "Spain",
        "Sweden",
        "Thailand",
        "Turkey",
    ], 
)
subset = subset[subset["Country"].isin(countries)]
### P2.3 ###


### P2.4 ###
# replace with st.selectbox
all_cancers = df["Cancer"].unique().tolist() 
cancer = st.selectbox(
    "Select Cancer Type",
    options=all_cancers, 
    index=all_cancers.index("Malignant neoplasm of stomach") 
)
subset = subset[subset["Cancer"] == cancer]
### P2.4 ###


### P2.5 ###
ages = [
    "Age <5",
    "Age 5-14",
    "Age 15-24",
    "Age 25-34",
    "Age 35-44",
    "Age 45-54",
    "Age 55-64",
    "Age >64",
]

age_selection = alt.selection_single(fields=["Age"], nearest=False, on="click", empty="none")

heatmap = alt.Chart(subset).mark_rect().encode(
    x=alt.X("Age", sort=ages, title="Age"),
    y=alt.Y("Country", title="Country"),
    color=alt.Color(
        "Rate",
        scale=alt.Scale(domain=[0.01, 1000], type="log", clamp=True),  
        legend=alt.Legend(title="Mortality Rate (log scale)")
    ),
    tooltip=["Country", "Age", "Rate"],
    # opacity=alt.condition(age_selection, alt.value(1), alt.value(0.3))
).add_selection(
    age_selection  
).properties(
    title=f"{cancer} Mortality Rates for {'Males' if sex == 'M' else 'Females'} in {year}",
    width=600,
    height=400
)

bar_chart = alt.Chart(subset).mark_bar().encode(
    x=alt.X("Pop", title="Population Size"),
    y=alt.Y("Country", title="Country", sort="-x"),
    tooltip=["Country", "Pop"]
).transform_filter(
    age_selection 
).properties(
    title=f"Population Size by Country for Selected Age Group",
    width=600,
    height=200
)

### P2.5 ###
st.altair_chart(heatmap & bar_chart, use_container_width=True)

countries_in_subset = subset["Country"].unique()
if len(countries_in_subset) != len(countries):
    if len(countries_in_subset) == 0:
        st.write("No data avaiable for given subset.")
    else:
        missing = set(countries) - set(countries_in_subset)
        st.write("No data available for " + ", ".join(missing) + ".")
