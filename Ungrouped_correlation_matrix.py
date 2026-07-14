# Install the necessary Excel engine
!pip install xlsxwriter -q

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
import io

# 1. Upload File
print("--- Step 1: Upload your input file ---")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[file_name]))

# 2. Data Preprocessing
cols_to_drop = [c for c in ['Host', 'Row', 'Dopant'] if c in df.columns]
numeric_data = df.drop(columns=cols_to_drop)

# 3. Correlation Analysis
print(f"\n--- Step 2: Calculating Correlation Matrix ---")
corr_matrix = numeric_data.corr()

print("\n--- Step 3: Correlation Matrix Table ---")
print(corr_matrix)

# 5. Generate Heatmaps
print("\n--- Step 4: Generating Graphs ---")

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, linewidths=0.5)
plt.title('Correlation Matrix (Physical Descriptors)', fontsize=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
            linewidths=0.5, annot_kws={"size": 18, "weight": "bold"})
plt.title('Correlation Matrix (Physical Descriptors)', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# 6. Export to Excel with Live Formulas
output_file = 'Correlation_Analysis_Ungrouped.xlsx'
num_features = len(corr_matrix)

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')

    workbook = writer.book
    worksheet = writer.sheets['Correlation_Matrix']

    header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
    formula_format = workbook.add_format({'bg_color': '#FDE9D9', 'num_format': '0.00'})

    worksheet.write(0, num_features + 1, 'Abs Corr vs DFT_E', header_format)
    dft_e_idx = list(corr_matrix.columns).index('DFT_E') + 1 
    dft_e_col_char = chr(ord('A') + dft_e_idx)

    for row_num in range(1, num_features + 1):
        cell_formula = f"=ABS({dft_e_col_char}{row_num + 1})"
        worksheet.write_formula(row_num, num_features + 1, cell_formula, formula_format)

print(f"\n--- Step 5: Downloading {output_file} ---")
files.download(output_file)