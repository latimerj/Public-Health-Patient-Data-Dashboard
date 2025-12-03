import pandas as pd


class HealthAnalyzer:
    """
    Simple analysis class for the synthetic Mental Health Patients dataset.
    Supports:
    - data cleaning
    - diagnosis frequency (Q1)
    - visits over time (Q2)
    - age and gender distribution (Q3)
    - treatment duration by diagnosis (Q4)
    - satisfaction score distribution (Q5)
    """

    def __init__(self, df: pd.DataFrame):
        # keep original and a working copy
        self.raw_df = df
        self.df = df.copy()

    def clean_data(self) -> pd.DataFrame:
        """
        Basic cleaning:
        - ensure Visit_Date is datetime
        - cast numeric columns
        - drop clearly invalid ages
        """
        # Convert date column
        if "Visit_Date" in self.df.columns:
            self.df["Visit_Date"] = pd.to_datetime(self.df["Visit_Date"], errors="coerce")

        # Numeric columns
        numeric_cols = [
            "Age",
            "Treatment_Duration_Weeks",
            "Num_Visits",
            "Satisfaction_Score",
            "Severity_Level",
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Drop impossible ages if present
        if "Age" in self.df.columns:
            self.df = self.df[(self.df["Age"] >= 0) & (self.df["Age"] <= 100)]

        # Drop rows missing critical fields (diagnosis or date)
        for col in ["Diagnosis", "Visit_Date"]:
            if col in self.df.columns:
                self.df = self.df.dropna(subset=[col])
        # Remove duplicates based on Patient_ID + Visit_Date (same visit shouldn't repeat)
        self.df = self.df.drop_duplicates(subset=["Patient_ID", "Visit_Date"])

        return self.df

    # Q1: Most common diagnoses
    def get_diagnosis_counts(self):
        if "Diagnosis" not in self.df.columns:
            return None
        return self.df["Diagnosis"].value_counts().sort_values(ascending=False)

    # Q2: Visits over time (by month)
    def get_monthly_visits(self):
        if "Visit_Date" not in self.df.columns:
            return None

        temp = self.df.dropna(subset=["Visit_Date"]).copy()
        temp["Month"] = temp["Visit_Date"].dt.to_period("M").dt.to_timestamp()
        return temp.groupby("Month").size()

    # Q3: Age and gender
    def get_age_series(self):
        if "Age" not in self.df.columns:
            return None
        return self.df["Age"].dropna()

    def get_gender_counts(self):
        if "Gender" not in self.df.columns:
            return None
        return self.df["Gender"].value_counts()

    # Q4: Treatment duration by diagnosis
    def get_duration_by_diagnosis(self):
        if "Diagnosis" not in self.df.columns or "Treatment_Duration_Weeks" not in self.df.columns:
            return None
        return (
            self.df
            .groupby("Diagnosis")["Treatment_Duration_Weeks"]
            .mean()
            .sort_values(ascending=False)
        )

    # Q5: Satisfaction score distribution
    def get_satisfaction_series(self):
        if "Satisfaction_Score" not in self.df.columns:
            return None
        return self.df["Satisfaction_Score"].dropna()