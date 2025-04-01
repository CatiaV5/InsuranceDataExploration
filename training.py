import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arff

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import XGBoost's regressor.
from xgboost import XGBRegressor

# Import the InsurancePreprocessor class from your module.
from preprocessor import InsurancePreprocessor


# ---------------------- Model Training Class ---------------------- #
class InsuranceModelTrainer:
    def __init__(self, preprocessor: InsurancePreprocessor,
                 test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the model trainer with a preprocessor instance.
        The preprocessor provides the preprocessed data and selected features.
        """
        self.preprocessor = preprocessor
        self.df_encoded = preprocessor.df_encoded
        self.selected_features = preprocessor.selected_features
        self.test_size = test_size
        self.random_state = random_state

    # ------------------ Random Forest Methods ------------------ #
    def train_rf_all_features(self):
        """
        Train a Random Forest regression model using all features (excluding ID and target).
        Reduced hyperparameter tuning is performed using GridSearchCV.
        Returns evaluation metrics and the best model.
        """
        print("Starting Random Forest training on ALL features...")
        start_time = time.time()
        X = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        param_grid = {
            'n_estimators': [50],
            'max_depth': [10, 20, 50],
            'min_samples_split': [3, 5],
            'min_samples_leaf': [2, 4],
            'random_state': [self.random_state]
        }
        rf = RandomForestRegressor()
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error',
                            n_jobs=-1, verbose=2)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        cv_results_df = pd.DataFrame(grid.cv_results_)
        cv_results_df.to_excel("cv_results_rf_all_features.xlsx", index=False)
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "val_mse": mean_squared_error(y_val, y_val_pred),
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "val_mae": mean_absolute_error(y_val, y_val_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "val_r2": r2_score(y_val, y_val_pred),
            "best_params": grid.best_params_
        }
        elapsed = time.time() - start_time
        print("[Random Forest ALL Features] Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print(f"Random Forest ALL features training completed in {elapsed:.2f} seconds.")
        return metrics, best_model

    # ------------------ Generalized Linear Model Methods ------------------ #
    # Using TweedieRegressor with a log-transformed target.
    def train_glm_all_features(self):
        """
        Train a Tweedie regression model using all features (excluding ID and target) with log-transformed target.
        Uses a pipeline with RobustScaler and fits on np.log1p(target).
        After prediction, applies np.expm1 to back-transform.
        Reduced hyperparameter tuning is performed using GridSearchCV.
        Returns evaluation metrics (on the original scale) and the best model.
        """
        print("Starting GLM (TweedieRegressor) training on ALL features with log-transformed target...")
        start_time = time.time()
        X = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('glm', TweedieRegressor(max_iter=1500, tol=1e-5))
        ])
        param_grid = {
            'glm__power': [1.0, 1.5, 1.7],
            'glm__alpha': [0.001, 0.01],
            'glm__max_iter': [1500],
            'glm__tol': [1e-5, 1e-4]
        }
        grid = GridSearchCV(pipeline, param_grid, cv=3,
                            scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        grid.fit(X_train, y_train_log)
        best_model = grid.best_estimator_
        cv_results_df = pd.DataFrame(grid.cv_results_)
        cv_results_df.to_excel("cv_results_glm_all_features.xlsx", index=False)
        y_train_pred_log = best_model.predict(X_train)
        y_val_pred_log = best_model.predict(X_val)
        y_train_pred = np.expm1(y_train_pred_log)
        y_val_pred = np.expm1(y_val_pred_log)
        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "val_mse": mean_squared_error(y_val, y_val_pred),
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "val_mae": mean_absolute_error(y_val, y_val_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "val_r2": r2_score(y_val, y_val_pred),
            "best_params": grid.best_params_
        }
        elapsed = time.time() - start_time
        print("[GLM ALL Features] Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print(f"GLM ALL features training completed in {elapsed:.2f} seconds.")
        return metrics, best_model

    # ------------------ XGBoost Methods with Early Stopping ------------------ #
    def train_xgb_all_features(self):
        """
        Train an XGBoost regression model using all features (excluding ID and target)
        with early stopping. A smaller hyperparameter grid is used.
        Returns evaluation metrics and the best model.
        """
        print("Starting XGBoost training on ALL features with early stopping...")
        start_time = time.time()
        X = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        # Smaller hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4],
            'learning_rate': [0.1, 0.05],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'random_state': [self.random_state]
        }
        xgb_model = XGBRegressor(objective='reg:squarederror', verbosity=1)
        # Pass early_stopping_rounds and eval_set to GridSearchCV via fit_params
        grid = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error',
                            n_jobs=-1, verbose=2)
        grid.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 early_stopping_rounds=10,
                 verbose=False)
        best_model = grid.best_estimator_
        cv_results_df = pd.DataFrame(grid.cv_results_)
        cv_results_df.to_excel("cv_results_xgb_all_features.xlsx", index=False)
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "val_mse": mean_squared_error(y_val, y_val_pred),
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "val_mae": mean_absolute_error(y_val, y_val_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "val_r2": r2_score(y_val, y_val_pred),
            "best_params": grid.best_params_
        }
        elapsed = time.time() - start_time
        print("[XGB ALL Features] Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print(f"XGBoost ALL features training completed in {elapsed:.2f} seconds.")
        return metrics, best_model



    # ------------------ Utility Methods ------------------ #
    @staticmethod
    def save_metrics(metrics_dict: dict, filename: str):
        """Save evaluation metrics to an Excel file."""
        df_metrics = pd.DataFrame(metrics_dict, index=[0])
        df_metrics.to_excel(filename, index=False)
        print(f"✅ Saved metrics to: {filename}")

    @staticmethod
    def plot_metrics(metrics_dict: dict, filename_prefix: str):
        """
        Plot evaluation metrics (MSE, MAE, and R² for both training and validation)
        and save the figure as a PNG.
        Here we use line plots (curves) connecting the train and validation values.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Plot Mean Squared Error.
        axes[0].plot(["Train MSE", "Validation MSE"],
                     [metrics_dict["train_mse"], metrics_dict["val_mse"]],
                     marker="o", linestyle="--", color="blue")
        axes[0].set_title("Mean Squared Error")
        axes[0].set_ylabel("MSE")
        # Plot Mean Absolute Error.
        axes[1].plot(["Train MAE", "Validation MAE"],
                     [metrics_dict["train_mae"], metrics_dict["val_mae"]],
                     marker="o", linestyle="--", color="orange")
        axes[1].set_title("Mean Absolute Error")
        # Plot R² Score.
        axes[2].plot(["Train R²", "Validation R²"],
                     [metrics_dict["train_r2"], metrics_dict["val_r2"]],
                     marker="o", linestyle="--", color="green")
        axes[2].set_title("R² Score")
        fig.tight_layout()
        filename = f"{filename_prefix}_metrics.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ Saved metrics plot to: {filename}")

    @staticmethod
    def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=3,
                            n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5),
                            filename_prefix="learning_curve"):
        """
        Generate and save a learning curve plot using scikit-learn's learning_curve.
        """
        train_sizes, train_scores, cv_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                              train_sizes=train_sizes, scoring='r2')
        train_scores_mean = np.mean(train_scores, axis=1)
        cv_scores_mean = np.mean(cv_scores, axis=1)
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color="red", label="Training score")
        plt.plot(train_sizes, cv_scores_mean, 'o-', color="green", label="Cross-validation score")
        plt.title(title)
        plt.xlabel("Training Examples")
        plt.ylabel("Score (R²)")
        plt.legend(loc="best")
        plt.grid(True)
        filename = f"{filename_prefix}.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ Saved learning curve plot to: {filename}")

    def run_training(self):
        """
        Run training for Random Forest, GLM, and XGBoost models on all features and selected features.
        Save evaluation metrics and plots to Excel/PNG, plot learning curves, and print all metrics.
        """
        # ------------------ Random Forest Training ------------------ #
        # Uncomment if you want to run RF training.
        print("\n=== Starting Random Forest Training ===")
        rf_metrics_all, rf_model_all = self.train_rf_all_features()
        self.save_metrics(rf_metrics_all, "metrics_rf_all_features.xlsx")
        self.plot_metrics(rf_metrics_all, "loss_rf_all_features")
        X_all = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y_all = self.df_encoded["target"]
        self.plot_learning_curve(rf_model_all, X_all, y_all, title="RF All Features Learning Curve", cv=3,
                                  filename_prefix="rf_all_learning_curve")

        #rf_metrics_sel, rf_model_sel = self.train_rf_selected_features()
        #self.save_metrics(rf_metrics_sel, "metrics_rf_selected_features.xlsx")
        #self.plot_metrics(rf_metrics_sel, "loss_rf_selected_features")
        #X_sel = self.df_encoded[self.selected_features]
        #self.plot_learning_curve(rf_model_sel, X_sel, y_all, title="RF Selected Features Learning Curve", cv=3,
        #                          filename_prefix="rf_selected_learning_curve")

        # ------------------ GLM (TweedieRegressor) Training ------------------ #
        # Uncomment if you want to run GLM training.
        # print("\n=== Starting GLM (TweedieRegressor) Training ===")
        # glm_metrics_all, glm_model_all = self.train_glm_all_features()
        # self.save_metrics(glm_metrics_all, "metrics_glm_all_features.xlsx")
        # self.plot_metrics(glm_metrics_all, "loss_glm_all_features")
        # self.plot_learning_curve(glm_model_all, X_all, y_all, title="GLM All Features Learning Curve", cv=3,
        #                          filename_prefix="glm_all_learning_curve")
        #
        # glm_metrics_sel, glm_model_sel = self.train_glm_selected_features()
        # self.save_metrics(glm_metrics_sel, "metrics_glm_selected_features.xlsx")
        # self.plot_metrics(glm_metrics_sel, "loss_glm_selected_features")
        # self.plot_learning_curve(glm_model_sel, X_sel, y_all, title="GLM Selected Features Learning Curve", cv=3,
        #                          filename_prefix="glm_selected_learning_curve")

        # ------------------ XGBoost Training ------------------ #
        print("\n=== Starting XGBoost Training ===")
        # Train using all features.
        xgb_metrics_all, xgb_model_all = self.train_xgb_all_features()
        self.save_metrics(xgb_metrics_all, "metrics_xgb_all_features.xlsx")
        self.plot_metrics(xgb_metrics_all, "loss_xgb_all_features")
        X_all = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y_all = self.df_encoded["target"]
        self.plot_learning_curve(xgb_model_all, X_all, y_all, title="XGB All Features Learning Curve", cv=3,
                                 filename_prefix="xgb_all_learning_curve")

        # Train using selected features.
        xgb_metrics_sel, xgb_model_sel = self.train_xgb_selected_features()
        self.save_metrics(xgb_metrics_sel, "metrics_xgb_selected_features.xlsx")
        self.plot_metrics(xgb_metrics_sel, "loss_xgb_selected_features")
        X_sel = self.df_encoded[self.selected_features]
        self.plot_learning_curve(xgb_model_sel, X_sel, y_all, title="XGB Selected Features Learning Curve", cv=3,
                                 filename_prefix="xgb_selected_learning_curve")

        print("\n=== All Model Training Completed ===")



