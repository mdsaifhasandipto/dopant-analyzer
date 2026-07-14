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
st.markdown("Upload your CSV to run full analyses (Appendix B + Correlations)")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is None:
    st.stop()

df_raw = pd.read_csv(uploaded_file)
st.dataframe(df_raw.head(), use_container_width=True)

if st.button("🚀 Run Full Analysis", type="primary"):
    with st.spinner("Running..."):
        # Data Preparation
        group_a = ['Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg']
        group_b = ['Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt']

        df_raw['Group'] = df_raw['Dopant'].apply(lambda x: 'A' if str(x).strip() in group_a else ('B' if str(x).strip() in group_b else 'Other'))
        df_clean = df_raw[df_raw['Group'] != 'Other'].copy()

        features = ['ed', 'Mag']
        target = 'DFT_E'
        df_model = df_clean.dropna(subset=features + [target]).copy()

        # Regression (Appendix B)
        def run_regression(sub_df):
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
            return {'model': model, 'y_pred': y_pred, 'y_loo': y_loo, 'y_actual': y.values, 'dopants': sub_df['Dopant'].values}

        res_a = run_regression(df_model[df_model['Group']=='A'])
        res_b = run_regression(df_model[df_model['Group']=='B'])

        y_act = np.concatenate([res_a['y_actual'], res_b['y_actual']])
        y_pre = np.concatenate([res_a['y_pred'], res_b['y_pred']])
        y_loo = np.concatenate([res_a['y_loo'], res_b['y_loo']])

        r2_train = r2_score(y_act, y_pre)
        mae_train = mean_absolute_error(y_act, y_pre)
        r2_loo = r2_score(y_act, y_loo)
        mae_loo = mean_absolute_error(y_act, y_loo)

        # 4-Panel Plot
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        def annotate(ax, x, y, labels):
            for i, txt in enumerate(labels):
                ax.annotate(txt, (x[i], y[i]), xytext=(5,5), textcoords='offset points', fontsize=8)

        # Panel 1 & 2
        axs[0,0].scatter(res_a['y_actual'], res_a['y_pred'], c='blue', label='Group A')
        axs[0,0].scatter(res_b['y_actual'], res_b['y_pred'], c='orange', label='Group B')
        annotate(axs[0,0], res_a['y_actual'], res_a['y_pred'], res_a['dopants'])
        annotate(axs[0,0], res_b['y_actual'], res_b['y_pred'], res_b['dopants'])
        axs[0,0].plot([y_act.min(), y_act.max()], [y_act.min(), y_act.max()], 'r--')
        axs[0,0].set_title(f'Training Parity (R²={r2_train:.3f})')
        axs[0,0].legend()

        axs[0,1].scatter(res_a['y_actual'], res_a['y_loo'], c='blue', marker='s', label='Group A LOOCV')
        axs[0,1].scatter(res_b['y_actual'], res_b['y_loo'], c='orange', marker='s', label='Group B LOOCV')
        annotate(axs[0,1], res_a['y_actual'], res_a['y_loo'], res_a['dopants'])
        annotate(axs[0,1], res_b['y_actual'], res_b['y_loo'], res_b['dopants'])
        axs[0,1].plot([y_act.min(), y_act.max()], [y_act.min(), y_act.max()], 'k--')
        axs[0,1].set_title(f'LOOCV (R²={r2_loo:.3f})')

        st.pyplot(fig)

        # Download Plot
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button("Download Full Plot", buf, "parity_plots.png", "image/png")

        # Correlations (fixed)
        st.subheader("Correlation Heatmaps")
        numeric_cols = df_model.select_dtypes(include=[np.number]).columns
        corr_data = df_model[numeric_cols].corr()

        fig_corr = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
        st.pyplot(fig_corr)

        # Excel Download
        excel_buf = BytesIO()
        with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
            pd.DataFrame({'Metric': ['R2_Train', 'MAE_Train', 'R2_LOOCV', 'MAE_LOOCV'],
                         'Value': [r2_train, mae_train, r2_loo, mae_loo]}).to_excel(writer, sheet_name='Summary', index=False)
            corr_data.to_excel(writer, sheet_name='Correlation')

        excel_buf.seek(0)
        st.download_button("Download Excel Report", excel_buf, "Full_Analysis_Report.xlsx", 
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.success("✅ Done! Use the download buttons above.")
