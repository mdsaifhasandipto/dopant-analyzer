import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from io import BytesIO

st.set_page_config(page_title="Dopant Analyzer", layout="wide")
st.title("🧪 Materials Dopant Analyzer")
st.markdown("**Upload CSV** to run all analyses (Appendix B, C, D)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file and st.button("Run All Analyses", type="primary"):
    df_raw = pd.read_csv(uploaded_file)
    # ... (Full combined logic similar to previous version - uses the same functions as your scripts)

    st.success("Analysis complete! Plots generated.")
    # Display plots and offer downloads