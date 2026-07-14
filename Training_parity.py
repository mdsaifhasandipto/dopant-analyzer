import sys
import subprocess
try:
    import xlsxwriter
except ModuleNotFoundError:
    print("Installing required package 'xlsxwriter'...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xlsxwriter"])
    import xlsxwriter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
import io
from google.colab import files

# ==========================================
# 1. INTERACTIVE ANY-FILE UPLOAD
# ==========================================
print("Please select your dataset file (CSV format):")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"Loaded File: {filename}")

df_raw = pd.read_csv(io.BytesIO(uploaded[filename]))
initial_count = len(df_raw)

# ==========================================
# 2. GROUP DEFINITIONS & PREPROCESSING
# ==========================================
group_a_elements = ['Sc', 'Y', 'La', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg']
group_b_elements = ['Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt']

df_raw['Group'] = df_raw['Dopant'].apply(lambda x: 'A' if x in group_a_elements else ('B' if x in group_b_elements else 'Other'))
df_filtered = df_raw[df_raw['Group'] != 'Other']

features = ['ed', 'Mag']
target = 'DFT_E'
df_clean = df_filtered.dropna(subset=features + [target]).reset_index(drop=True)
final_count = len(df_clean)

print(f"\nPreprocessing Summary:")
print(f" -> Initial Rows: {initial_count}")
print(f" -> Active Rows After Grouping & NaN Dropping: {final_count} (Dropped: {initial_count - final_count})")

# ==========================================
# 3. STATISTICAL ANALYSIS REGRESSION PIPELINE
# ==========================================
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
        
    influence = OLSInfluence(model)
    leverage = influence.hat_matrix_diag
    resid_sq = model.resid ** 2
    mse = model.mse_resid
    
    return {
        'model': model, 
        'vif': vif_data, 
        'y_pred': y_pred, 
        'y_loo': y_loo, 
        'y_actual': y, 
        'dopants': sub_df['Dopant'].reset_index(drop=True),
        'sub_df': sub_df.reset_index(drop=True),
        'leverage': leverage,
        'resid_sq': resid_sq,
        'mse': mse
    }

df_a = df_clean[df_clean['Group'] == 'A'].copy()
df_b = df_clean[df_clean['Group'] == 'B'].copy()

res_a = run_group_regression(df_a)
res_b = run_group_regression(df_b)

# ==========================================
# 4. EXPLICIT COMPREHENSIVE OUTPUT REPORTING
# ==========================================
print("\n" + "="*70 + "\nDETAILED REGRESSION & STATISTICAL METRICS REPORT\n" + "="*70)

for res, g_label in [(res_a, "GROUP A (Early/Late TMs)"), (res_b, "GROUP B (Mid TMs)")]:
    m = res['model']
    print(f"\n--- {g_label} ---")
    print(f"Isolated Equation: E_ads = ({m.params['const']:.3f}) + ({m.params['ed']:.3f} * ed) + ({m.params['Mag']:.3f} * Mag)")
    print(f"F-Statistic (Overall Fit Test): {m.fvalue:.3f} (p-value: {m.f_pvalue:.4e})")
    print(f"Degrees of Freedom (Residuals): {int(m.df_resid)}")
    print("\nParameter-Specific Tests (t-test & Confidence Intervals):")
    
    for idx, term in enumerate(['const', 'ed', 'Mag']):
        print(f"  [{term}]:")
        print(f"    Coefficient:     {m.params[term]:.4f}")
        print(f"    t-stat:          {m.tvalues[term]:.3f} (p-value: {m.pvalues[term]:.4e})")
        print(f"    95% Conf. Int.:  [{m.conf_int().iloc[idx, 0]:.4f}, {m.conf_int().iloc[idx, 1]:.4f}]")
        if term != 'const':
            print(f"    Variance Inf. F: {res['vif'][idx]:.2f}")

y_all_act = np.concatenate([res_a['y_actual'], res_b['y_actual']])
y_all_pre = np.concatenate([res_a['y_pred'], res_b['y_pred']])
y_all_loo = np.concatenate([res_a['y_loo'], res_b['y_loo']])

global_r2_train = r2_score(y_all_act, y_all_pre)
global_mae_train = mean_absolute_error(y_all_act, y_all_pre)
global_r2_loo = r2_score(y_all_act, y_all_loo)
global_mae_loo = mean_absolute_error(y_all_act, y_all_loo)

print("\n" + "="*70 + "\nAGGREGATED SYSTEM VALIDATION METRICS\n" + "="*70)
print(f"Global Training Fit:  R² = {global_r2_train:.3f} | MAE = {global_mae_train:.3f} eV")
print(f"Global Validation LOOCV: R² = {global_r2_loo:.3f} | MAE = {global_mae_loo:.3f} eV")

# ==========================================
# 5. GENERATE 4-PANEL MASTER DIAGNOSTIC FIG.
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

def label_scatter_points(ax, x_vals, y_vals, element_names):
    for i, name in enumerate(element_names):
        x_val = x_vals.iloc[i] if hasattr(x_vals, 'iloc') else x_vals[i]
        y_val = y_vals.iloc[i] if hasattr(y_vals, 'iloc') else y_vals[i]
        ax.annotate(name, (x_val, y_val), xytext=(4, 4), 
                    textcoords='offset points', fontsize=8, alpha=0.7)

axs[0, 0].scatter(res_a['y_actual'], res_a['y_pred'], c='#3498db', s=75, edgecolors='k', alpha=0.85, label='Group A (Early/Late TMs)')
axs[0, 0].scatter(res_b['y_actual'], res_b['y_pred'], c='#e67e22', s=75, edgecolors='k', alpha=0.85, label='Group B (Mid TMs)')
label_scatter_points(axs[0, 0], res_a['y_actual'], res_a['y_pred'], res_a['dopants'])
label_scatter_points(axs[0, 0], res_b['y_actual'], res_b['y_pred'], res_b['dopants'])
axs[0, 0].plot([y_all_act.min(), y_all_act.max()], [y_all_act.min(), y_all_act.max()], 'r--', lw=2)
axs[0, 0].set_title(f'1: Training Parity (\( R^2 = {global_r2_train:.3f} \) | MAE = {global_mae_train:.3f} eV)', fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel('DFT Calculated \( E_{ads} \) (eV)', fontsize=10)
axs[0, 0].set_ylabel('Model Predicted \( E_{ads} \) (eV)', fontsize=10)
axs[0, 0].legend(loc='upper left')
axs[0, 0].grid(True, alpha=0.3)

axs[0, 1].scatter(res_a['y_actual'], res_a['y_loo'], c='#3498db', marker='s', s=65, edgecolors='k', alpha=0.85, label='Group A (LOOCV)')
axs[0, 1].scatter(res_b['y_actual'], res_b['y_loo'], c='#e67e22', marker='s', s=65, edgecolors='k', alpha=0.85, label='Group B (LOOCV)')
label_scatter_points(axs[0, 1], res_a['y_actual'], res_a['y_loo'], res_a['dopants'])
label_scatter_points(axs[0, 1], res_b['y_actual'], res_b['y_loo'], res_b['dopants'])
axs[0, 1].plot([y_all_act.min(), y_all_act.max()], [y_all_act.min(), y_all_act.max()], 'k--', alpha=0.6, lw=1.5)
axs[0, 1].set_title(f'2: LOOCV Validation Plot (\( R^2 = {global_r2_loo:.3f} \) | MAE = {global_mae_loo:.3f} eV)', fontsize=12, fontweight='bold')
axs[0, 1].set_xlabel('DFT Calculated \( E_{ads} \) (eV)', fontsize=10)
axs[0, 1].set_ylabel('LOOCV Predicted \( E_{ads} \) (eV)', fontsize=10)
axs[0, 1].legend(loc='upper left')
axs[0, 1].grid(True, alpha=0.3)

axs[1, 0].scatter(res_a['model'].fittedvalues, res_a['model'].resid, c='#3498db', s=65, edgecolors='k', alpha=0.8)
label_scatter_points(axs[1, 0], res_a['model'].fittedvalues, res_a['model'].resid, res_a['dopants'])
axs[1, 0].axhline(0, color='red', linestyle='--')
axs[1, 0].set_title('3: Residuals vs Fitted (Group A)', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('Fitted \( E_{ads} \) (eV)', fontsize=10)
axs[1, 0].set_ylabel('Residual (eV)', fontsize=10)
axs[1, 0].grid(True, alpha=0.3)

axs[1, 1].scatter(res_b['model'].fittedvalues, res_b['model'].resid, c='#e67e22', s=65, edgecolors='k', alpha=0.8)
label_scatter_points(axs[1, 1], res_b['model'].fittedvalues, res_b['model'].resid, res_b['dopants'])
axs[1, 1].axhline(0, color='red', linestyle='--')
axs[1, 1].set_title('4: Residuals vs Fitted (Group B)', fontsize=12, fontweight='bold')
axs[1, 1].set_xlabel('Fitted \( E_{ads} \) (eV)', fontsize=10)
axs[1, 1].set_ylabel('Residual (eV)', fontsize=10)
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Bilinear_Parity_Masterplots.png', dpi=300)
plt.show()

# ==========================================
# 6. EXCEL LIVE-FORMULA EXPORT SYSTEM
# ==========================================
excel_filename = 'Bilinear_Model_Report.xlsx'
wb = xlsxwriter.Workbook(excel_filename, {'nan_inf_to_errors': True})

fmt_h = wb.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1, 'align': 'center'})
fmt_d = wb.add_format({'border': 1, 'align': 'center'})
fmt_f = wb.add_format({'bg_color': '#FFF2CC', 'border': 1, 'align': 'center'})

len_a = len(df_a)
len_b = len(df_b)

ws_sum = wb.add_worksheet('Analysis Summary')
ws_sum.write(0, 0, 'Statistical Parameters Summary (HC3 Robust Error Coefficients)', fmt_h)
ws_sum.write_row(2, 0, ['Group', 'Term', 'Coefficient', 'P-Value', 'VIF (Intercept Omitted)'], fmt_h)

def write_summary_block(ws, start_row, g_name, r_obj):
    ws.write(start_row, 0, g_name, fmt_d)
    terms = ['const', 'ed', 'Mag']
    for idx, t in enumerate(terms):
        r = start_row + idx
        ws.write(r, 1, t, fmt_d)
        ws.write(r, 2, r_obj['model'].params[t], fmt_d)
        ws.write(r, 3, r_obj['model'].pvalues[t], fmt_d)
        ws.write(r, 4, r_obj['vif'][idx] if t != 'const' else 'N/A', fmt_d)

write_summary_block(ws_sum, 3, 'Group A', res_a)
write_summary_block(ws_sum, 7, 'Group B', res_b)

ws1 = wb.add_worksheet('Panel 1 - Training Parity')
ws1.write_row(0, 0, ['Dopant Element', 'Group Designation', 'Descriptor ed', 'Descriptor Mag', 'DFT Calculated E_ads', 'Model Predicted E_ads', 'Absolute Error', 'Root Mean Squared Error (RMSE)'], fmt_h)

for i in range(final_count):
    row_idx = i + 1
    if i < len_a:
        ws1.write_row(row_idx, 0, [res_a['dopants'][i], 'A', res_a['sub_df']['ed'].iloc[i], res_a['sub_df']['Mag'].iloc[i], res_a['y_actual'].iloc[i]], fmt_d)
        ws1.write_formula(row_idx, 5, f"='Analysis Summary'!$C$4 + ('Analysis Summary'!$C$5 * C{row_idx+1}) + ('Analysis Summary'!$C$6 * D{row_idx+1})", fmt_f)
        ws1.write_formula(row_idx, 6, f"=ABS(F{row_idx+1}-E{row_idx+1})", fmt_f)
        ws1.write_formula(row_idx, 7, f"=SQRT(SUMSQ(F2:F{len_a+1}-E2:E{len_a+1})/COUNTA(A2:A{len_a+1}))", fmt_f)
    else:
        b_idx = i - len_a
        ws1.write_row(row_idx, 0, [res_b['dopants'][b_idx], 'B', res_b['sub_df']['ed'].iloc[b_idx], res_b['sub_df']['Mag'].iloc[b_idx], res_b['y_actual'].iloc[b_idx]], fmt_d)
        ws1.write_formula(row_idx, 5, f"='Analysis Summary'!$C$8 + ('Analysis Summary'!$C$9 * C{row_idx+1}) + ('Analysis Summary'!$C$10 * D{row_idx+1})", fmt_f)
        ws1.write_formula(row_idx, 6, f"=ABS(F{row_idx+1}-E{row_idx+1})", fmt_f)
        ws1.write_formula(row_idx, 7, f"=SQRT(SUMSQ(F{len_a+2}:F{final_count+1}-E{len_a+2}:E{final_count+1})/COUNTA(A{len_a+2}:A{final_count+1}))", fmt_f)

ws2 = wb.add_worksheet('Panel 2 - LOOCV Parity')
ws2.write_row(0, 0, ['Dopant Element', 'Group Designation', 'DFT Calculated E_ads', 'LOOCV Predicted E_ads'], fmt_h)

for i in range(final_count):
    row_idx = i + 1
    if i < len_a:
        ws2.write_row(row_idx, 0, [res_a['dopants'][i], 'A', res_a['y_actual'].iloc[i], res_a['y_loo'][i]], fmt_d)
    else:
        b_idx = i - len_a
        ws2.write_row(row_idx, 0, [res_b['dopants'][b_idx], 'B', res_b['y_actual'].iloc[b_idx], res_b['y_loo'][b_idx]], fmt_d)

ws3 = wb.add_worksheet('Panel 3 - Group A Residuals')
ws3.write_row(0, 0, ['Dopant', 'Fitted Value (X)', 'Residual (Y)'], fmt_h)
for i in range(len_a):
    ws3.write(i + 1, 0, res_a['dopants'][i], fmt_d)
    ws3.write_formula(i + 1, 1, f"='Panel 1 - Training Parity'!F{i+2}", fmt_d)
    ws3.write_formula(i + 1, 2, f"='Panel 1 - Training Parity'!E{i+2}-'Panel 1 - Training Parity'!F{i+2}", fmt_f)

ws4 = wb.add_worksheet('Panel 4 - Group B Residuals')
ws4.write_row(0, 0, ['Dopant', 'Fitted Value (X)', 'Residual (Y)'], fmt_h)
for i in range(len_b):
    ws4.write(i + 1, 0, res_b['dopants'][i], fmt_d)
    ws4.write_formula(i + 1, 1, f"='Panel 1 - Training Parity'!F{len_a+i+2}", fmt_d)
    ws4.write_formula(i + 1, 2, f"='Panel 1 - Training Parity'!E{len_a+i+2}-'Panel 1 - Training Parity'!F{len_a+i+2}", fmt_f)

wb.close()
print("\nExecution complete! Plots saved as 'Bilinear_Parity_Masterplots.png' and live-formula workbook built.")

print("Triggering automatic file download browser prompt...")
files.download(excel_filename)