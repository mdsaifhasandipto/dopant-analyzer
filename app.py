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
import xlsxwriter

st.set_page_config(page_title="Dopant Analyzer", layout="wide")
st.title("🧪 Materials Dopant Analyzer")
st.markdown("Upload your CSV file to run Appendix B, C, and D analyses")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload your CSV file")
    st.stop()

df_raw = pd.read_csv(uploaded_file)
st.success(f"Loaded {len(df_raw)} rows")

if st.button("🚀 Run All Analyses", type="primary"):
    with st.spinner("Running regression, plots and correlations..."):
        # ===================== Appendix B Logic =====================
        group_a_elements = ['Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg']
        group_b_elements = ['Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt']

        df_raw['Group'] = df_raw['Dopant'].apply(lambda x: 'A' if str(x) in group_a_elements else ('B' if str(x) in group_b_elements else 'Other'))
        df_filtered = df_raw[df_raw['Group'] != 'Other'].copy()

        features = ['ed', 'Mag']
        target = 'DFT_E'
        df_clean = df_filtered.dropna(subset=features + [target]).reset_index(drop=True)

        def run_group_regression(sub_df):
            X = sub_df[features]
            y = sub_df[target]
            X_sm = sm.add_constant(X)
            model = sm.OLS(y, X_sm).fit(cov_type='HC3')
            sk_model = LinearRegression().fit(X, y)
            y_pred = sk_model.predict(X)
            
            loo = LeaveOneOut()
            y_loo = np.zeros(len(y))
            for train_idx, test_idx in loo.split(X):
                m = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
                y_loo[test_idx] = m.predict(X.iloc[test_idx])
            
            return {
                'model': model,
                'y_pred': y_pred,
                'y_loo': y_loo,
                'y_actual': y.values,
                'dopants': sub_df['Dopant'].values
            }

        df_a = df_clean[df_clean['Group'] == 'A'].copy()
        df_b = df_clean[df_clean['Group'] == 'B'].copy()
        res_a = run_group_regression(df_a)
        res_b = run_group_regression(df_b)

        y_all_act = np.concatenate([res_a['y_actual'], res_b['y_actual']])
        y_all_pre = np.concatenate([res_a['y_pred'], res_b['y_pred']])
        y_all_loo = np.concatenate([res_a['y_loo'], res_b['y_loo']])

        global_r2_train = r2_score(y_all_act, y_all_pre)
        global_mae_train = mean_absolute_error(y_all_act, y_all_pre)
        global_r2_loo = r2_score(y_all_act, y_all_loo)
        global_mae_loo = mean_absolute_error(y_all_act, y_all_loo)

        # 4-Panel Plot
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        def label_points(ax, x, y, names):
            for i, name in enumerate(names):
                ax.annotate(name, (x[i], y[i]), xytext=(4,4), textcoords='offset points', fontsize=8, alpha=0.7)

        # Panel 1
        axs[0,0].scatter(res_a['y_actual'], res_a['y_pred'], c='#3498db', s=75, edgecolors='k', label='Group A')
        axs[0,0].scatter(res_b['y_actual'], res_b['y_pred'], c='#e67e22', s=75, edgecolors='k', label='Group B')
        label_points(axs[0,0], res_a['y_actual'], res_a['y_pred'], res_a['dopants'])
        label_points(axs[0,0], res_b['y_actual'], res_b['y_pred'], res_b['dopants'])
        axs[0,0].plot([y_all_act.min(), y_all_act.max()], [y_all_act.min(), y_all_act.max()], 'r--')
        axs[0,0].set_title(f'Training Parity (R² = {global_r2_train:.3f})')
        axs[0,0].legend()

        st.pyplot(fig)

        # Download Plot
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        st.download_button("📥 Download Master Parity Plot", buf, "Bilinear_Parity_Masterplots.png", "image/png")

        # ===================== Correlation Plots (C & D) =====================
        st.subheader("Correlation Matrices")
        cols_to_drop = ['Host', 'Dopant', 'Row']
        numeric_data = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])

        fig_corr = plt.figure(figsize=(10,8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='RdBu_r', center=0)
        st.pyplot(fig_corr)

        st.success("Analysis complete!")

        # Simple Excel Download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame({'R2_Train': [global_r2_train], 'MAE_Train': [global_mae_train]}).to_excel(writer, sheet_name='Metrics')
        output.seek(0)
        st.download_button("📥 Download Excel Report", output, "Analysis_Report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
