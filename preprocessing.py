import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arff


def prepare_data_for_modeling(df_freq, df_sev):
    df_sev_agg = df_sev.groupby('IDpol')['ClaimAmount'].sum().reset_index()
    df_sev_agg.rename(columns={'ClaimAmount': 'total_claim_amount'}, inplace=True)
    df_merged = pd.merge(df_freq, df_sev_agg, on='IDpol', how='left')
    df_merged['total_claim_amount'] = df_merged['total_claim_amount'].fillna(0)
    df_merged['target'] = 0
    mask = df_merged['Exposure'] > 0
    df_merged.loc[mask, 'target'] = (df_merged.loc[mask, 'total_claim_amount'] /
                                     df_merged.loc[mask, 'Exposure'])
    return df_merged


def one_hot_encode(df, categorical_cols):
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def run_pca_on_features(df, features_to_exclude=None, n_components=5):
    if features_to_exclude is None:
        features_to_exclude = []
    df_for_pca = df.drop(columns=features_to_exclude, errors='ignore')
    # Include one-hot encoded columns which may be 'uint8' or 'bool'
    allowed_types = ['int64', 'float64', 'bool', 'uint8']
    pca_features = df_for_pca.select_dtypes(include=allowed_types).columns
    df_for_pca = df_for_pca[pca_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_for_pca)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca, pca_features


def load_arff_file(filename: str, custom_columns: list[str] = None) -> pd.DataFrame:
    data_arff = arff.load(open(filename, 'r'))
    records = data_arff['data']
    attributes = data_arff['attributes']
    column_names = [attr[0] for attr in attributes] if custom_columns is None else custom_columns
    df = pd.DataFrame(records, columns=column_names)
    return df


######## LOAD IN ARFF ######################

freq_cols = [
    "IDpol", "ClaimNb", "Exposure", "Area", "VehPower",
    "VehAge", "DrivAge", "BonusMalus", "VehBrand",
    "VehGas", "Density", "Region"
]

df_freq = load_arff_file("Data/freMTPL2freq.arff", custom_columns=freq_cols)
sev_cols = ["IDpol", "ClaimAmount"]
df_sev = load_arff_file("Data/freMTPL2sev.arff", custom_columns=sev_cols)

###############################################

# 1) Prepare and merge
df_combined = prepare_data_for_modeling(df_freq, df_sev)

# 2) Decide which columns are categorical
categorical_cols = ["Area", "VehBrand", "VehGas", "Region"]

# 3) One-hot encode the categorical columns
df_encoded = one_hot_encode(df_combined, categorical_cols)

# 4) Run PCA (Exclude IDpol, 'target', 'ClaimNb', etc.)
exclude_from_pca = ["IDpol", "target", "ClaimNb", "Exposure", "total_claim_amount"]
pca_model, X_pca, pca_features = run_pca_on_features(df_encoded, features_to_exclude=exclude_from_pca, n_components=5)

# 1. Save explained variance ratios
explained_variance_df = pd.DataFrame({
    'Component': [f'PC{i}' for i in range(1, len(pca_model.explained_variance_ratio_) + 1)],
    'ExplainedVarianceRatio': pca_model.explained_variance_ratio_
})

# 2. Save cumulative explained variance plot
cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
plt.figure()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plot_filename = "pca_cumulative_variance.png"
plt.savefig(plot_filename)
plt.close()

# 3. Save PCA loadings (contribution of each original feature to each component)
# Use the pca_features returned from run_pca_on_features
loadings = pd.DataFrame(
    pca_model.components_.T,
    columns=[f"PC{i}" for i in range(1, pca_model.n_components_ + 1)],
    index=pca_features
)

# --- Additional Data: Number of features and Top Features per PC ---

num_features_used = len(pca_features)
metadata_df = pd.DataFrame({'TotalFeaturesUsed': [num_features_used]})

top_features_per_pc = pd.DataFrame()
for pc in loadings.columns:
    sorted_features = loadings[pc].abs().sort_values(ascending=False)
    top_features = sorted_features.head(3).index.tolist()
    top_features_per_pc[pc] = top_features

# 4. Save everything to Excel
output_excel_path = "pca_output_detailed.xlsx"
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    explained_variance_df.to_excel(writer, sheet_name='ExplainedVariance', index=False)
    loadings.to_excel(writer, sheet_name='PCALoadings')
    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    top_features_per_pc.to_excel(writer, sheet_name='TopFeaturesPerPC', index=False)

print(f"✅ Saved PCA explained variance plot to: {plot_filename}")
print(f"✅ Saved PCA results to: {output_excel_path}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arff

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def prepare_data_for_modeling(df_freq, df_sev):
    df_sev_agg = df_sev.groupby('IDpol')['ClaimAmount'].sum().reset_index()
    df_sev_agg.rename(columns={'ClaimAmount': 'total_claim_amount'}, inplace=True)
    df_merged = pd.merge(df_freq, df_sev_agg, on='IDpol', how='left')
    df_merged['total_claim_amount'] = df_merged['total_claim_amount'].fillna(0)
    df_merged['target'] = 0
    mask = df_merged['Exposure'] > 0
    df_merged.loc[mask, 'target'] = (df_merged.loc[mask, 'total_claim_amount'] /
                                     df_merged.loc[mask, 'Exposure'])
    return df_merged


def one_hot_encode(df, categorical_cols):
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


def run_pca_on_features(df, features_to_exclude=None, n_components=5):
    if features_to_exclude is None:
        features_to_exclude = []
    df_for_pca = df.drop(columns=features_to_exclude, errors='ignore')
    # Include one-hot encoded columns which may be 'uint8' or 'bool'
    allowed_types = ['int64', 'float64', 'bool', 'uint8']
    pca_features = df_for_pca.select_dtypes(include=allowed_types).columns
    df_for_pca = df_for_pca[pca_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_for_pca)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca, pca_features


def load_arff_file(filename: str, custom_columns: list[str] = None) -> pd.DataFrame:
    data_arff = arff.load(open(filename, 'r'))
    records = data_arff['data']
    attributes = data_arff['attributes']
    column_names = [attr[0] for attr in attributes] if custom_columns is None else custom_columns
    df = pd.DataFrame(records, columns=column_names)
    return df


######## LOAD IN ARFF ######################

freq_cols = [
    "IDpol", "ClaimNb", "Exposure", "Area", "VehPower",
    "VehAge", "DrivAge", "BonusMalus", "VehBrand",
    "VehGas", "Density", "Region"
]

df_freq = load_arff_file("freMTPL2freq.arff", custom_columns=freq_cols)
sev_cols = ["IDpol", "ClaimAmount"]
df_sev = load_arff_file("freMTPL2sev.arff", custom_columns=sev_cols)

###############################################

# 1) Prepare and merge
df_combined = prepare_data_for_modeling(df_freq, df_sev)

# 2) Decide which columns are categorical
categorical_cols = ["Area", "VehBrand", "VehGas", "Region"]

# 3) One-hot encode the categorical columns
df_encoded = one_hot_encode(df_combined, categorical_cols)

# 4) Run PCA (Exclude IDpol, 'target', 'ClaimNb', etc.)
exclude_from_pca = ["IDpol", "target", "ClaimNb", "Exposure", "total_claim_amount"]
pca_model, X_pca, pca_features = run_pca_on_features(df_encoded, features_to_exclude=exclude_from_pca, n_components=5)

# 1. Save explained variance ratios
explained_variance_df = pd.DataFrame({
    'Component': [f'PC{i}' for i in range(1, len(pca_model.explained_variance_ratio_) + 1)],
    'ExplainedVarianceRatio': pca_model.explained_variance_ratio_
})

# 2. Save cumulative explained variance plot
cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
plt.figure()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plot_filename = "pca_cumulative_variance.png"
plt.savefig(plot_filename)
plt.close()

# 3. Save PCA loadings (contribution of each original feature to each component)
# Use the pca_features returned from run_pca_on_features
loadings = pd.DataFrame(
    pca_model.components_.T,
    columns=[f"PC{i}" for i in range(1, pca_model.n_components_ + 1)],
    index=pca_features
)

# --- Additional Data: Number of features and Top Features per PC ---

num_features_used = len(pca_features)
metadata_df = pd.DataFrame({'TotalFeaturesUsed': [num_features_used]})

top_features_per_pc = pd.DataFrame()
for pc in loadings.columns:
    sorted_features = loadings[pc].abs().sort_values(ascending=False)
    top_features = sorted_features.head(3).index.tolist()
    top_features_per_pc[pc] = top_features

# 4. Save everything to Excel
output_excel_path = "pca_output_detailed.xlsx"
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    explained_variance_df.to_excel(writer, sheet_name='ExplainedVariance', index=False)
    loadings.to_excel(writer, sheet_name='PCALoadings')
    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    top_features_per_pc.to_excel(writer, sheet_name='TopFeaturesPerPC', index=False)

print(f"✅ Saved PCA explained variance plot to: {plot_filename}")
print(f"✅ Saved PCA results to: {output_excel_path}")
