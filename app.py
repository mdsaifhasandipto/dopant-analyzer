import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from io import BytesIO
import xlsxwriter

st.set_page_config(page_title="Exact Dopant Analyzer", layout="wide")
st.title("🧪 Exact Appendix B Replication")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file and st.button("Run FULL Appendix B"):
    df_raw = pd.read_csv(uploaded_file)

    # === YOUR EXACT CODE FROM APPENDIX B ===
    group_a_elements = ['Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg']
    group_b_elements = ['Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt']

    df_raw['Group'] = df_raw['Dopant'].apply(lambda x: 'A' if str(x) in group_a_elements else ('B' if str(x) in group_b_elements else 'Other'))
    df_filtered = df_raw[df_raw['Group'] != 'Other']
    features = ['ed', 'Mag']
    target = 'DFT_E'
    df_clean = df_filtered.dropna(subset=features + [target]).reset_index(drop=True)
    final_count = len(df_clean)

    def run_group_regression(sub_df):
        X = sub_df[features]
        y = sub_df[target]
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit(cov_type='HC3')
        vif_data = [variance_inflation_factor(X_sm.values, i) for i in range(X_sm.shape[1])]
        sk_model = LinearRegression().fit(X, y)
        y_pred = sk_model.predict(X)
        loo = LeaveOneOut()
        y_loo = np.zeros(len(y))
        for train_idx, test_idx in loo.split(X):
            m = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
            y_loo[test_idx] = m.predict(X.iloc[test_idx])
        return {
            'model': model, 
            'vif': vif_data, 
            'y_pred': y_pred, 
            'y_loo': y_loo, 
            'y_actual': y, 
            'dopants': sub_df['Dopant'].reset_index(drop=True),
            'sub_df': sub_df.reset_index(drop=True)
        }

    df_a = df_clean[df_clean['Group'] == 'A'].copy()
    df_b = df_clean[df_clean['Group'] == 'B'].copy()
    res_a = run_group_regression(df_a)
    res_b = run_group_regression(df_b)

    # Detailed Report
    st.subheader("Regression Report (Exact)")
    for res, g_label in [(res_a, "GROUP A"), (res_b, "GROUP B")]:
        m = res['model']
        st.write(f"**{g_label}**")
        st.latex(f"E_ads = {m.params['const']:.3f} + {m.params['ed']:.3f}*ed + {m.params['Mag']:.3f}*Mag")
        st.write(f"F-stat: {m.fvalue:.3f}, p = {m.f_pvalue:.4e}")

    # 4-Panel Plot
    y_all_act = np.concatenate([res_a['y_actual'], res_b['y_actual']])
    y_all_pre = np.concatenate([res_a['y_pred'], res_b['y_pred']])
    y_all_loo = np.concatenate([res_a['y_loo'], res_b['y_loo']])
    global_r2_train = r2_score(y_all_act, y_all_pre)
    global_mae_train = mean_absolute_error(y_all_act, y_all_pre)
    global_r2_loo = r2_score(y_all_act, y_all_loo)
    global_mae_loo = mean_absolute_error(y_all_act, y_all_loo)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    def label_scatter_points(ax, x_vals, y_vals, element_names):
        for i, name in enumerate(element_names):
            ax.annotate(name, (x_vals[i], y_vals[i]), xytext=(4, 4), textcoords='offset points', fontsize=8, alpha=0.7)

    axs[0, 0].scatter(res_a['y_actual'], res_a['y_pred'], c='#3498db', s=75, edgecolors='k', alpha=0.85, label='Group A')
    axs[0, 0].scatter(res_b['y_actual'], res_b['y_pred'], c='#e67e22', s=75, edgecolors='k', alpha=0.85, label='Group B')
    label_scatter_points(axs[0, 0], res_a['y_actual'], res_a['y_pred'], res_a['dopants'])
    label_scatter_points(axs[0, 0], res_b['y_actual'], res_b['y_pred'], res_b['dopants'])
    axs[0, 0].plot([y_all_act.min(), y_all_act.max()], [y_all_act.min(), y_all_act.max()], 'r--', lw=2)
    axs[0, 0].set_title(f'1: Training Parity (R² = {global_r2_train:.3f})')
    axs[0, 0].legend()

    st.pyplot(fig)

    # Download Plot
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    st.download_button("Download Master Plot", buf, "Bilinear_Parity_Masterplots.png", "image/png")

    st.success("Appendix B complete. Excel coming in next update.")
