import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from LatimerJasmine_MentalHealthProject import HealthAnalyzer

st.title("Public Health Patient Data Dashboard")

st.write(
    "Upload mental health dataset to explore diagnoses, visits over time, "
    "demographics, treatment duration, and satisfaction scores."
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload the mental health patients CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully.")

    # Preview of the dataset
    st.write("### Dataset Preview")
    st.subheader("First Few Rows")
    st.dataframe(df.head())
    st.subheader("Last Few Rows")
    st.dataframe(df.tail())

    # Create analyzer and clean data
    base_analyzer = HealthAnalyzer(df)
    clean_df = base_analyzer.clean_data()

    st.write("### Cleaned Data Sample")
    st.dataframe(clean_df.head())

    # INTERACTIVE FILTERS
    st.sidebar.header("Filters")

    # Age filter
    if "Age" in clean_df.columns and not clean_df["Age"].dropna().empty:
        min_age = int(clean_df["Age"].min())
        max_age = int(clean_df["Age"].max())
        age_range = st.sidebar.slider(
            "Age range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )
    else:
        age_range = None

    # Gender filter
    if "Gender" in clean_df.columns:
        gender_options = sorted(clean_df["Gender"].dropna().unique())
        selected_genders = st.sidebar.multiselect(
            "Gender",
            options=gender_options,
            default=gender_options
        )
    else:
        selected_genders = None

    # Diagnosis filter 
    if "Diagnosis" in clean_df.columns:
        diagnosis_options = sorted(clean_df["Diagnosis"].dropna().unique())
        selected_diagnoses = st.sidebar.multiselect(
            "Diagnosis (Department or Clinical Area)",
            options=diagnosis_options,
            default=diagnosis_options
        )
    else:
        selected_diagnoses = None

    # Apply filters
    filtered_df = clean_df.copy()

    if age_range is not None:
        filtered_df = filtered_df[
            (filtered_df["Age"] >= age_range[0]) &
            (filtered_df["Age"] <= age_range[1])
        ]

    if selected_genders is not None and len(selected_genders) > 0:
        filtered_df = filtered_df[filtered_df["Gender"].isin(selected_genders)]

    if selected_diagnoses is not None and len(selected_diagnoses) > 0:
        filtered_df = filtered_df[filtered_df["Diagnosis"].isin(selected_diagnoses)]

    st.write("### Filtered Data Sample")
    if filtered_df.empty:
        st.warning("No records match the selected filters.")
    else:
        st.dataframe(filtered_df.head())

    # Use the filtered data for analysis
    analyzer = HealthAnalyzer(filtered_df)

    # Tabs for each of the 5 project questions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Q1: Diagnoses",
        "Q2: Visits Over Time",
        "Q3: Demographics",
        "Q4: Treatment Duration",
        "Q5: Satisfaction Scores"
    ])

    #  Q1: Most common diagnoses 
    with tab1:
        st.subheader("Question 1: What are the most common mental health diagnoses?")

        diag_counts = analyzer.get_diagnosis_counts()

        if diag_counts is None or diag_counts.empty:
            st.error("Diagnosis data is missing after filtering.")
        else:
            st.write("Diagnosis counts:")
            st.dataframe(
                diag_counts.reset_index().rename(
                    columns={"index": "Diagnosis", "Diagnosis": "Count"}
                )
            )

            fig1, ax1 = plt.subplots()
            ax1.bar(diag_counts.index, diag_counts.values)
            ax1.set_xlabel("Diagnosis")
            ax1.set_ylabel("Number of Patients")
            ax1.set_title("Most Common Mental Health Diagnoses")
            plt.xticks(rotation=45)
            st.pyplot(fig1)

            st.subheader("Interpretation")
            top_dx = diag_counts.idxmax()
            top_count = diag_counts.max()
            st.write(
                f"- {top_dx} appears most often in the filtered dataset, with {top_count} patients."
            )
            st.write(
                "This helps identify the most common conditions for the selected age, gender, and diagnosis filters."
            )

    #  Q2: Visits over time (monthly) 
    with tab2:
        st.subheader("Question 2: How many mental health visits occur over time?")

        monthly_visits = analyzer.get_monthly_visits()

        if monthly_visits is None or monthly_visits.empty:
            st.error("Visit_Date data is missing or invalid after filtering.")
        else:
            st.write("Monthly number of visits:")
            st.dataframe(
                monthly_visits.reset_index().rename(
                    columns={0: "Number of Visits"}
                )
            )

            fig2, ax2 = plt.subplots()
            ax2.plot(monthly_visits.index, monthly_visits.values, marker="o")
            ax2.set_xlabel("Month")
            ax2.set_ylabel("Number of Visits")
            ax2.set_title("Monthly Mental Health Visits")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)

            st.subheader("Interpretation")
            peak_month = monthly_visits.idxmax()
            peak_value = monthly_visits.max()
            st.write(
                f"- The highest number of visits occurs in {peak_month.strftime('%Y-%m')} "
                f"with {peak_value} visits for the selected filters."
            )
            st.write(
                "This trend line shows how demand for mental health services changes over time "
                "among the filtered patient group."
            )

    # Q3: Demographics (age & gender) 
    with tab3:
        st.subheader("Question 3: What is the age and gender distribution?")

        ages = analyzer.get_age_series()
        gender_counts = analyzer.get_gender_counts()

        # Age histogram
        if ages is not None and not ages.empty:
            fig3, ax3 = plt.subplots()
            ax3.hist(ages, bins=10)
            ax3.set_xlabel("Age")
            ax3.set_ylabel("Number of Patients")
            ax3.set_title("Age Distribution of Patients")
            st.pyplot(fig3)
        else:
            st.error("Age data is missing after filtering.")

        # Gender bar chart
        if gender_counts is not None and not gender_counts.empty:
            st.write("Gender counts:")
            st.dataframe(
                gender_counts.reset_index().rename(
                    columns={"index": "Gender", "Gender": "Count"}
                )
            )

            fig4, ax4 = plt.subplots()
            ax4.bar(gender_counts.index.astype(str), gender_counts.values)
            ax4.set_xlabel("Gender")
            ax4.set_ylabel("Number of Patients")
            ax4.set_title("Gender Distribution of Patients")
            st.pyplot(fig4)
        else:
            st.error("Gender data is missing after filtering.")

        st.subheader("Interpretation")
        st.write(
            "The age histogram shows which age ranges are most common within the filtered group."
        )
        st.write(
            "The gender chart shows which gender groups are most represented among patients under the selected filters."
        )

    # Q4: Treatment duration by diagnosis
    with tab4:
        st.subheader("Question 4: How does treatment duration differ by diagnosis?")

        duration_by_dx = analyzer.get_duration_by_diagnosis()

        if duration_by_dx is None or duration_by_dx.empty:
            st.error("Treatment duration or diagnosis data is missing after filtering.")
        else:
            st.write("Average treatment duration (weeks) by diagnosis:")
            st.dataframe(
                duration_by_dx.reset_index().rename(
                    columns={"Treatment_Duration_Weeks": "Average Duration (weeks)"}
                )
            )

            fig5, ax5 = plt.subplots()
            ax5.bar(duration_by_dx.index, duration_by_dx.values)
            ax5.set_xlabel("Diagnosis")
            ax5.set_ylabel("Average Treatment Duration (weeks)")
            ax5.set_title("Average Treatment Duration by Diagnosis")
            plt.xticks(rotation=45)
            st.pyplot(fig5)

            st.subheader("Interpretation")
            top_dx_dur = duration_by_dx.idxmax()
            top_dur = duration_by_dx.max()
            st.write(
                f"- {top_dx_dur} has the longest average treatment duration at about {top_dur:.1f} weeks "
                "for the current filter selection."
            )
            st.write(
                "This suggests that some conditions require longer or more intensive care among the filtered patients."
            )

    # Q5: Satisfaction score distribution 
    with tab5:
        st.subheader("Question 5: What is the distribution of satisfaction scores?")

        satisfaction_series = analyzer.get_satisfaction_series()

        if satisfaction_series is None or satisfaction_series.empty:
            st.error("Satisfaction_Score data is missing after filtering.")
        else:
            fig6, ax6 = plt.subplots()
            ax6.hist(satisfaction_series, bins=[1, 2, 3, 4, 5, 6], align="left", rwidth=0.8)
            ax6.set_xlabel("Satisfaction Score (1 to 5)")
            ax6.set_ylabel("Number of Patients")
            ax6.set_title("Satisfaction Score Distribution")
            st.pyplot(fig6)

            st.subheader("Interpretation")
            most_common_score = satisfaction_series.value_counts().idxmax()
            count_most_common = satisfaction_series.value_counts().max()
            st.write(
                f"- The most common satisfaction score is {int(most_common_score)} "
                f"with {count_most_common} patients in the filtered data."
            )
            st.write(
                "If most scores are 4 or 5, this suggests that patients in the selected group are generally satisfied with their care."
            )

else:
    st.info("Please upload the mental health patients CSV file to begin.")
