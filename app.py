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
import io
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name

# Set page configuration
st.set_page_config(page_title="Catalysis Data Analytics Dashboard", layout="wide")
st.title("🔬 Catalysis Materials Informatics Platform")
st.markdown("Welcome, Examiners. Please select an analytical pipeline below, upload your dataset, and view the live statistical models, figures, and spreadsheet exports.")

# Create four distinct pipeline views using tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Appendix B: Bilinear Training & LOOCV", 
    "🔀 Appendix C: Grouped Pearson Correlation", 
    "🌐 Appendix D: Ungrouped Pearson Correlation",
    "🌋 Appendix E: HER Volcano Plot (Nørskov Model)"
])

# ==========================================
# PIPELINE 1: APPENDIX B
# ==========================================
with tab1:
    st.header("Bilinear Regression & Leave-One-Out Cross-Validation (LOOCV)")
    file_b = st.file_uploader("Upload Dataset for Appendix B (CSV Format)", type=["csv"], key="file_b")
    
    if file_b is not None:
        df_raw = pd.read_csv(file_b)
        initial_count = len(df_raw)

        group_a_elements = ['Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg']
        group_b_elements = ['Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt']

        df_raw['Group'] = df_raw['Dopant'].apply(lambda x: 'A' if x in group_a_elements else ('B' if x in group_b_elements else 'Other'))
        df_filtered = df_raw[df_raw['Group'] != 'Other']

        features = ['ed', 'Mag']
        target = 'DFT_E'
        df_clean = df_filtered.dropna(subset=features + [target]).reset_index(drop=True)
        final_count = len(df_clean)

        st.success(f"Preprocessing Complete: Initial Rows ({initial_count}) ➔ Active Rows ({final_count}). Dropped: {initial_count - final_count}")

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
                'model': model, 'vif': vif_data, 'y_pred': y_pred, 'y_loo': y_loo, 
                'y_actual': y, 'dopants': sub_df['Dopant'].reset_index(drop=True),
                'sub_df': sub_df.reset_index(drop=True)
            }

        df_a = df_clean[df_clean['Group'] == 'A'].copy()
        df_b = df_clean[df_clean['Group'] == 'B'].copy()

        res_a = run_group_regression(df_a)
        res_b = run_group_regression(df_b)

        st.markdown("### Detailed Regression & Statistical Metrics Report")
        col1, col2 = st.columns(2)
        
        for res, g_label, col in [(res_a, "GROUP A (Early/Late TMs)", col1), (res_b, "GROUP B (Mid TMs)", col2)]:
            with col:
                m = res['model']
                st.markdown(f"#### {g_label}")
                st.code(f"E_ads = ({m.params['const']:.3f}) + ({m.params['ed']:.3f} * ed) + ({m.params['Mag']:.3f} * Mag)")
                st.write(f"**F-Statistic:** {m.fvalue:.3f} (p-value: {m.f_pvalue:.4e})")
                st.write(f"**Degrees of Freedom (Residuals):** {int(m.df_resid)}")
                
                param_df = pd.DataFrame({
                    "Coefficient": m.params,
                    "t-stat": m.tvalues,
                    "p-value": m.pvalues,
                    "VIF": [np.nan] + [res['vif'][1], res['vif'][2]]
                })
                st.dataframe(param_df.style.format("{:.4e}"))

        y_all_act = np.concatenate([res_a['y_actual'], res_b['y_actual']])
        y_all_pre = np.concatenate([res_a['y_pred'], res_b['y_pred']])
        y_all_loo = np.concatenate([res_a['y_loo'], res_b['y_loo']])

        st.markdown("### Aggregated System Validation Metrics")
        st.info(f"**Global Training Fit:** R² = {r2_score(y_all_act, y_all_pre):.3f} | MAE = {mean_absolute_error(y_all_act, y_all_pre):.3f} eV")
        st.info(f"**Global Validation LOOCV:** R² = {r2_score(y_all_act, y_all_loo):.3f} | MAE = {mean_absolute_error(y_all_act, y_all_loo):.3f} eV")

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        def label_pts(ax, x, y, labels):
            for idx, name in enumerate(labels):
                ax.annotate(name, (x.iloc[idx], y[idx] if isinstance(y, np.ndarray) else y.iloc[idx]), xytext=(4, 4), textcoords='offset points', fontsize=8, alpha=0.7)

        axs[0, 0].scatter(res_a['y_actual'], res_a['y_pred'], c='#3498db', s=75, edgecolors='k', label='Group A')
        axs[0, 0].scatter(res_b['y_actual'], res_b['y_pred'], c='#e67e22', s=75, edgecolors='k', label='Group B')
        label_pts(axs[0, 0], res_a['y_actual'], res_a['y_pred'], res_a['dopants'])
        label_pts(axs[0, 0], res_b['y_actual'], res_b['y_pred'], res_b['dopants'])
        axs[0, 0].plot([y_all_act.min(), y_all_act.max()], [y_all_act.min(), y_all_act.max()], 'r--', lw=2)
        axs[0, 0].set_title('1: Training Parity', fontweight='bold')
        
        axs[0, 1].scatter(res_a['y_actual'], res_a['y_loo'], c='#3498db', marker='s', s=65, edgecolors='k', label='Group A (LOOCV)')
        axs[0, 1].scatter(res_b['y_actual'], res_b['y_loo'], c='#e67e22', marker='s', s=65, edgecolors='k', label='Group B (LOOCV)')
        label_pts(axs[0, 1], res_a['y_actual'], res_a['y_loo'], res_a['dopants'])
        label_pts(axs[0, 1], res_b['y_actual'], res_b['y_loo'], res_b['dopants'])
        axs[0, 1].plot([y_all_act.min(), y_all_act.max()], [y_all_act.min(), y_all_act.max()], 'k--', lw=1.5)
        axs[0, 1].set_title('2: LOOCV Validation Plot', fontweight='bold')

        axs[1, 0].scatter(res_a['model'].fittedvalues, res_a['model'].resid, c='#3498db', s=65, edgecolors='k')
        label_pts(axs[1, 0], res_a['model'].fittedvalues, res_a['model'].resid, res_a['dopants'])
        axs[1, 0].axhline(0, color='red', linestyle='--')
        axs[1, 0].set_title('3: Residuals vs Fitted (Group A)', fontweight='bold')

        axs[1, 1].scatter(res_b['model'].fittedvalues, res_b['model'].resid, c='#e67e22', s=65, edgecolors='k')
        label_pts(axs[1, 1], res_b['model'].fittedvalues, res_b['model'].resid, res_b['dopants'])
        axs[1, 1].axhline(0, color='red', linestyle='--')
        axs[1, 1].set_title('4: Residuals vs Fitted (Group B)', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        output_b = io.BytesIO()
        wb = xlsxwriter.Workbook(output_b, {'nan_inf_to_errors': True})
        fmt_h = wb.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1, 'align': 'center'})
        fmt_d = wb.add_format({'border': 1, 'align': 'center'})
        fmt_f = wb.add_format({'bg_color': '#FFF2CC', 'border': 1, 'align': 'center'})

        ws_sum = wb.add_worksheet('Analysis Summary')
        ws_sum.write(0, 0, 'Statistical Parameters Summary', fmt_h)
        ws_sum.write_row(2, 0, ['Group', 'Term', 'Coefficient', 'P-Value', 'VIF'], fmt_h)
        
        def write_summary_block(ws, start_row, g_name, r_obj):
            ws.write(start_row, 0, g_name, fmt_d)
            for idx, t in enumerate(['const', 'ed', 'Mag']):
                r = start_row + idx
                ws.write(r, 1, t, fmt_d)
                ws.write(r, 2, r_obj['model'].params[t], fmt_d)
                ws.write(r, 3, r_obj['model'].pvalues[t], fmt_d)
                ws.write(r, 4, r_obj['vif'][idx] if t != 'const' else 'N/A', fmt_d)

        write_summary_block(ws_sum, 3, 'Group A', res_a)
        write_summary_block(ws_sum, 7, 'Group B', res_b)

        ws1 = wb.add_worksheet('Panel 1 - Training Parity')
        ws1.write_row(0, 0, ['Dopant Element', 'Group Designation', 'Descriptor ed', 'Descriptor Mag', 'DFT Calculated E_ads', 'Model Predicted E_ads', 'Absolute Error'], fmt_h)
        
        len_a = len(df_a)
        for i in range(final_count):
            row_idx = i + 1
            if i < len_a:
                ws1.write_row(row_idx, 0, [res_a['dopants'][i], 'A', res_a['sub_df']['ed'].iloc[i], res_a['sub_df']['Mag'].iloc[i], res_a['y_actual'].iloc[i]], fmt_d)
                ws1.write_formula(row_idx, 5, f"='Analysis Summary'!$C$4 + ('Analysis Summary'!$C$5 * C{row_idx+1}) + ('Analysis Summary'!$C$6 * D{row_idx+1})", fmt_f)
            else:
                b_idx = i - len_a
                ws1.write_row(row_idx, 0, [res_b['dopants'][b_idx], 'B', res_b['sub_df']['ed'].iloc[b_idx], res_b['sub_df']['Mag'].iloc[b_idx], res_b['y_actual'].iloc[b_idx]], fmt_d)
                ws1.write_formula(row_idx, 5, f"='Analysis Summary'!$C$8 + ('Analysis Summary'!$C$9 * C{row_idx+1}) + ('Analysis Summary'!$C$10 * D{row_idx+1})", fmt_f)
            ws1.write_formula(row_idx, 6, f"=ABS(F{row_idx+1}-E{row_idx+1})", fmt_f)

        wb.close()
        st.download_button(label="📥 Download Bilinear Model Report (Excel)", data=output_b.getvalue(), file_name="Bilinear_Model_Report.xlsx", mime="application/vnd.ms-excel")

# ==========================================
# PIPELINE 2: APPENDIX C
# ==========================================
with tab2:
    st.header("Grouped Pearson Correlation Matrices")
    file_c = st.file_uploader("Upload Dataset for Appendix C (CSV Format)", type=["csv"], key="file_c")
    
    if file_c is not None:
        df_c = pd.read_csv(file_c)
        gB = ['Cr', 'Mo', 'Mn', 'Tc', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt']

        df_g1 = df_c[df_c['Dopant'].isin(gB)].copy()
        df_g2 = df_c[~df_c['Dopant'].isin(gB)].copy()

        cols_to_drop = ['Host', 'Dopant', 'Row']
        df_g1_num = df_g1.drop(columns=[c for c in cols_to_drop if c in df_g1.columns])
        df_g2_num = df_g2.drop(columns=[c for c in cols_to_drop if c in df_g2.columns])

        corr_g1 = df_g1_num.corr()
        corr_g2 = df_g2_num.corr()

        st.markdown("### Correlation Matrix Visualizations")
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.subheader("Group B (Mid TM Dopants)")
            fig_c1, ax_c1 = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_g1, annot=True, cmap='RdBu_r', center=0, fmt='.2f', linewidths=0.5, ax=ax_c1)
            st.pyplot(fig_c1)
            
        with col_c2:
            st.subheader("Group A (Early/Late TM Dopants)")
            fig_c2, ax_c2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_g2, annot=True, cmap='RdBu_r', center=0, fmt='.2f', linewidths=0.5, ax=ax_c2)
            st.pyplot(fig_c2)

        output_c = io.BytesIO()
        with pd.ExcelWriter(output_c, engine='xlsxwriter') as writer:
            corr_g1.to_excel(writer, sheet_name='Group_B')
            corr_g2.to_excel(writer, sheet_name='Group_A')
        st.download_button(label="📥 Download Grouped Correlation Report", data=output_c.getvalue(), file_name="Grouped_Correlation_Analysis.xlsx", mime="application/vnd.ms-excel")

# ==========================================
# PIPELINE 3: APPENDIX D
# ==========================================
with tab3:
    st.header("Ungrouped Pearson Correlation System")
    file_d = st.file_uploader("Upload Dataset for Appendix D (CSV Format)", type=["csv"], key="key_d")
    
    if file_d is not None:
        df_d = pd.read_csv(file_d)
        cols_to_drop = [c for c in ['Host', 'Row', 'Dopant'] if c in df_d.columns]
        numeric_data = df_d.drop(columns=cols_to_drop)
        corr_matrix = numeric_data.corr()

        st.markdown("### Global Correlation Matrix Table")
        st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu_r', axis=None).format("{:.2f}"))

        st.markdown("### High-Resolution Analytical Heatmap")
        fig_d, ax_d = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f', linewidths=0.5, annot_kws={"size": 12, "weight": "bold"}, ax=ax_d)
        st.pyplot(fig_d)

        output_d = io.BytesIO()
        num_features = len(corr_matrix)
        
        with pd.ExcelWriter(output_d, engine='xlsxwriter') as writer:
            corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
            workbook  = writer.book
            worksheet = writer.sheets['Correlation_Matrix']

            header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            formula_format = workbook.add_format({'bg_color': '#FDE9D9', 'num_format': '0.00'})

            worksheet.write(0, num_features + 1, 'Abs Corr vs DFT_E', header_format)
            dft_e_idx = list(corr_matrix.columns).index('DFT_E') + 1 
            dft_e_col_char = chr(ord('A') + dft_e_idx)

            for row_num in range(1, num_features + 1):
                cell_formula = f"=ABS({dft_e_col_char}{row_num + 1})"
                worksheet.write_formula(row_num, num_features + 1, cell_formula, formula_format)

        st.download_button(label="📥 Download Ungrouped Live-Formula Workbook", data=output_d.getvalue(), file_name="Correlation_Analysis_Ungrouped.xlsx", mime="application/vnd.ms-excel")

# ==========================================
# PIPELINE 4: NEW APPENDIX E (HER VOLCANO PLOT)
# ==========================================
with tab4:
    st.header("HER Volcano Plot (Nørskov 2005 Model)")
    file_e = st.file_uploader("Upload Dataset for Appendix E (CSV Format, must contain 'DFT_E' and 'Dopant')", type=["csv"], key="file_e")
    
    if file_e is not None:
        df_e = pd.read_csv(file_e)
        
        if 'DFT_E' in df_e.columns and 'Dopant' in df_e.columns:
            kB = 8.61733326e-5   
            T = 298.15           
            k0 = 200.0           
            kBT = kB * T
            ln10 = np.log(10)

            def norskov_eq12_and_14(dG):
                exp_term = np.exp(-dG / kBT)
                log_i0 = np.log10(k0) - np.log10(1 + exp_term)
                if dG > 0:
                    log_i0 -= (dG / (kBT * ln10))
                return log_i0

            df_e['log_i0'] = df_e['DFT_E'].apply(norskov_eq12_and_14)

            fig_e, ax_e = plt.subplots(figsize=(11, 7))
            
            dG_range = np.linspace(-1.2, 1.2, 500)
            log_i0_theory = [norskov_eq12_and_14(x) for x in dG_range]
            ax_e.plot(dG_range, log_i0_theory, 'k--', lw=2.2, alpha=0.8, label='Theoretical Volcano (Eqs. 12 & 14)')

            ax_e.scatter(df_e['DFT_E'], df_e['log_i0'], color='royalblue', edgecolors='black', s=90, zorder=5, label='Calculated Sites')

            for idx, row in df_e.iterrows():
                ax_e.text(row['DFT_E'] + 0.02, row['log_i0'] + 0.02, str(row['Dopant']), fontsize=9)

            ax_e.axvline(0, color='red', linestyle='--', alpha=0.7, lw=1.5, label=r'Optimal $\Delta G_{H^*} \approx 0$ eV')
            ax_e.set_xlabel(r'$\Delta G_{H^*}$ (eV)', fontsize=14)
            ax_e.set_ylabel(r'$\log_{10} |j_0|$ (a.u.)', fontsize=14)
            ax_e.set_title('HER Volcano Plot', fontsize=15)
            ax_e.legend(fontsize=12)
            ax_e.grid(True, alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(fig_e)

            output_e = io.BytesIO()
            writer_e = pd.ExcelWriter(output_e, engine='xlsxwriter')
            df_e.to_excel(writer_e, index=False, sheet_name='HER_Analysis')

            workbook_e = writer_e.book
            worksheet_e = writer_e.sheets['HER_Analysis']

            worksheet_e.write('Z1', 'kBT (eV)')
            worksheet_e.write('AA1', kBT)
            worksheet_e.write('Z2', 'k0 (const)')
            worksheet_e.write('AA2', k0)
            worksheet_e.write('Z3', 'ln(10)')
            worksheet_e.write('AA3', ln10)

            dG_col = xl_col_to_name(df_e.columns.get_loc('DFT_E'))
            log_col = xl_col_to_name(df_e.columns.get_loc('log_i0'))

            for i in range(2, len(df_e) + 2):
                formula = (
                    f'=IF({dG_col}{i}<=0, '
                    f'LOG10($AA$2) - LOG10(1 + EXP(-{dG_col}{i}/$AA$1)), '
                    f'LOG10($AA$2) - LOG10(1 + EXP(-{dG_col}{i}/$AA$1)) - ({dG_col}{i}/($AA$1*$AA$3))'
                    ')'
                )
                worksheet_e.write_formula(f'{log_col}{i}', formula)

            writer_e.close()
            
            st.download_button(label="📥 Download Nørskov Volcano Spreadsheet Data", data=output_e.getvalue(), file_name="Norskov_volcano_data.xlsx", mime="application/vnd.ms-excel")
        else:
            st.error("Uploaded CSV missing required matching columns: 'DFT_E' and 'Dopant'.")
