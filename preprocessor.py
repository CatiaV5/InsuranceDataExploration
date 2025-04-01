import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arff

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class InsurancePreprocessor:
    def __init__(self, freq_filename: str, sev_filename: str,
                 freq_cols: list[str], sev_cols: list[str],
                 categorical_cols: list[str], exclude_from_pca: list[str],
                 pca_n_components: int = 5):
        """
        Initialize the preprocessor by loading ARFF files, preparing data,
        one-hot encoding, running PCA, and saving PCA results.
        """
        self.freq_filename = freq_filename
        self.sev_filename = sev_filename
        self.freq_cols = freq_cols
        self.sev_cols = sev_cols
        self.categorical_cols = categorical_cols
        self.exclude_from_pca = exclude_from_pca
        self.pca_n_components = pca_n_components

        # Load frequency and severity data from ARFF files.
        self.df_freq = self.load_arff_file(self.freq_filename, custom_columns=self.freq_cols)
        self.df_sev = self.load_arff_file(self.sev_filename, custom_columns=self.sev_cols)

        # Prepare and merge data.
        self.df_combined = self.prepare_data_for_modeling(self.df_freq, self.df_sev)
        # One-hot encode categorical columns.
        self.df_encoded = self.one_hot_encode(self.df_combined, self.categorical_cols)
        # Run PCA on encoded data.
        self.pca_model, self.X_pca, self.pca_features = self.run_pca_on_features(
            self.df_encoded, features_to_exclude=self.exclude_from_pca, n_components=self.pca_n_components
        )
        # Save PCA results to Excel and plot.
        self.save_pca_results()
        # Determine selected features based on top loadings per PC.
        self.selected_features = self._get_selected_features()

    def load_arff_file(self, filename: str, custom_columns: list[str] = None) -> pd.DataFrame:
        data_arff = arff.load(open(filename, 'r'))
        records = data_arff['data']
        attributes = data_arff['attributes']
        column_names = [attr[0] for attr in attributes] if custom_columns is None else custom_columns
        df = pd.DataFrame(records, columns=column_names)
        return df

    def prepare_data_for_modeling(self, df_freq: pd.DataFrame, df_sev: pd.DataFrame) -> pd.DataFrame:
        """
        Merges frequency and severity dataframes and creates the target variable.
        Target = total_claim_amount / Exposure.
        """
        df_sev_agg = df_sev.groupby('IDpol')['ClaimAmount'].sum().reset_index()
        df_sev_agg.rename(columns={'ClaimAmount': 'total_claim_amount'}, inplace=True)
        df_merged = pd.merge(df_freq, df_sev_agg, on='IDpol', how='left')
        df_merged['total_claim_amount'] = df_merged['total_claim_amount'].fillna(0)
        df_merged['target'] = 0
        mask = df_merged['Exposure'] > 0
        df_merged.loc[mask, 'target'] = (df_merged.loc[mask, 'total_claim_amount'] /
                                         df_merged.loc[mask, 'Exposure'])
        return df_merged

    def one_hot_encode(self, df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
        """Perform one-hot encoding on the specified categorical columns."""
        return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    def run_pca_on_features(self, df: pd.DataFrame, features_to_exclude: list[str] = None,
                            n_components: int = 5):
        """
        Runs PCA on the dataframe after dropping non-feature columns.
        One-hot encoded columns are included by allowing certain data types.
        Returns the fitted PCA, PCA-transformed data, and list of features used.
        """
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

    def save_pca_results(self):
        """Save PCA explained variance, loadings, metadata, and top features to Excel and the cumulative variance plot as PNG."""
        # Explained variance ratios.
        explained_variance_df = pd.DataFrame({
            'Component': [f'PC{i}' for i in range(1, len(self.pca_model.explained_variance_ratio_) + 1)],
            'ExplainedVarianceRatio': self.pca_model.explained_variance_ratio_
        })
        # Cumulative explained variance plot.
        cumulative_variance = np.cumsum(self.pca_model.explained_variance_ratio_)
        plt.figure()
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        plt.title("Cumulative Explained Variance by PCA Components")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.grid(True)
        plot_filename = "pca_cumulative_variance.png"
        plt.savefig(plot_filename)
        plt.close()
        # PCA loadings.
        loadings = pd.DataFrame(
            self.pca_model.components_.T,
            columns=[f"PC{i}" for i in range(1, self.pca_model.n_components_ + 1)],
            index=self.pca_features
        )
        # Metadata: number of features used.
        num_features_used = len(self.pca_features)
        metadata_df = pd.DataFrame({'TotalFeaturesUsed': [num_features_used]})
        # Top features per PC (top 3 by absolute loading).
        top_features_per_pc = pd.DataFrame()
        for pc in loadings.columns:
            sorted_features = loadings[pc].abs().sort_values(ascending=False)
            top_features = sorted_features.head(3).index.tolist()
            top_features_per_pc[pc] = top_features
        # Save everything to Excel.
        output_excel_path = "pca_output_detailed.xlsx"
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            explained_variance_df.to_excel(writer, sheet_name='ExplainedVariance', index=False)
            loadings.to_excel(writer, sheet_name='PCALoadings')
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            top_features_per_pc.to_excel(writer, sheet_name='TopFeaturesPerPC', index=False)
        print(f"✅ Saved PCA explained variance plot to: {plot_filename}")
        print(f"✅ Saved PCA results to: {output_excel_path}")

    def _get_selected_features(self) -> list[str]:
        """
        Extract a unique list of top features (top 3 per PC) from the PCA loadings.
        These are used for training a model on selected features.
        """
        loadings = pd.DataFrame(
            self.pca_model.components_.T,
            columns=[f"PC{i}" for i in range(1, self.pca_model.n_components_ + 1)],
            index=self.pca_features
        )
        top_features = set()
        for pc in loadings.columns:
            sorted_features = loadings[pc].abs().sort_values(ascending=False)
            top_features.update(sorted_features.head(3).index.tolist())
        return list(top_features)