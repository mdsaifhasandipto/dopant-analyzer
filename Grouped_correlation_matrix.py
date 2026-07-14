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

# 2. Define Groups and Preprocess
gB = ['Cr', 'Mo', 'Mn', 'Tc', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt']

df_g1 = df[df['Dopant'].isin(gB)].copy()
df_g2 = df[\~df['Dopant'].isin(gB)].copy()

cols_to_drop = ['Host', 'Dopant', 'Row']
df_g1_num = df_g1.drop(columns=[c for c in cols_to_drop if c in df_g1.columns])
df_g2_num = df_g2.drop(columns=[c for c in cols_to_drop if c in df_g2.columns])

corr_g1 = df_g1_num.corr()
corr_g2 = df_g2_num.corr()

# 4. Generate the 4 Heatmaps
print("\n--- Step 2: Generating Correlation Plots ---")

def plot_heatmap(corr, title, annot=False):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=annot, cmap='RdBu_r', center=0, fmt='.2f',
                linewidths=0.5, annot_kws={"size": 12, "weight": "bold"})
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

plot_heatmap(corr_g1, 'Correlation Matrix: Group B (No Values)', annot=False)
plot_heatmap(corr_g1, 'Correlation Matrix: Group B (With Values)', annot=True)

plot_heatmap(corr_g2, 'Correlation Matrix: Group A (No Values)', annot=False)
plot_heatmap(corr_g2, 'Correlation Matrix: Group A (With Values)', annot=True)

# 5. Export to Excel
output_file = 'Grouped_Correlation_Analysis.xlsx'

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    corr_g1.to_excel(writer, sheet_name='Group_B')
    corr_g2.to_excel(writer, sheet_name='Group A')

print(f"\n--- Step 3: Downloading {output_file} ---")
files.download(output_file)