import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arff
from sklearn.model_selection import learning_curve, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import XGBoost's regressor.
from xgboost import XGBRegressor

# Import TensorFlow and Keras.
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from sklearn.base import BaseEstimator



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
        print("Starting Random Forest training on ALL features...")
        start_time = time.time()
        X = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
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
    def train_glm_all_features(self):
        print("Starting GLM (TweedieRegressor) training on ALL features with log-transformed target...")
        start_time = time.time()
        X = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
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
        print("Starting XGBoost training on ALL features with early stopping...")
        start_time = time.time()
        X = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4],
            'learning_rate': [0.1, 0.05],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'random_state': [self.random_state]
        }
        xgb_model = XGBRegressor(objective='reg:squarederror', verbosity=1)
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

    # ------------------ Neural Network Methods ------------------ #
    def build_nn_model(self, input_dim):
        # Improved NN using an explicit Input layer, deeper architecture, and a lower learning rate.
        inputs = Input(shape=(input_dim,))
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.1)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(8, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(4, activation='relu')(x)

        outputs = Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse', metrics=['mae'])
        return model



    def train_nn_all_features(self):
        """
        Train a neural network model using all features (excluding ID and target).
        Normalizes the data using StandardScaler.
        Uses early stopping to halt training when validation loss stops improving.
        Plots training and validation loss curves, saves and prints evaluation metrics.
        Returns evaluation metrics and the trained model.
        """
        print("Starting Neural Network training on ALL features...")
        start_time = time.time()
        X = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model = self.build_nn_model(input_dim=X_train_scaled.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train_scaled, y_train,
                            validation_data=(X_val_scaled, y_val),
                            epochs=100, batch_size=32,
                            callbacks=[early_stop], verbose=2)
        train_loss, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
        val_loss, val_mae = model.evaluate(X_val_scaled, y_val, verbose=0)
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_r2": train_r2,
            "val_r2": val_r2
        }
        elapsed = time.time() - start_time
        print("[NN ALL Features] Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        plt.figure(figsize=(8,6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Neural Network ALL Features Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.savefig("nn_all_loss_curve.png")
        plt.close()
        print(f"Neural Network ALL features training completed in {elapsed:.2f} seconds.")
        self.save_metrics(metrics, "metrics_nn_all_features.xlsx")
        self.plot_learning_curve(model, X_train_scaled, y_train, title="NN All Features Learning Curve", cv=3,
                                 filename_prefix="nn_all_learning_curve")
        return metrics, model

    def train_nn_selected_features(self):
        """
        Train a neural network model using the selected features (derived from PCA).
        Normalizes the data using StandardScaler.
        Uses early stopping and plots training and validation loss curves.
        Returns evaluation metrics (on the original scale) and the trained model.
        """
        if not self.selected_features:
            raise ValueError("No selected features available from PCA.")
        print("Starting Neural Network training on SELECTED features...")
        start_time = time.time()
        X = self.df_encoded[self.selected_features]
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model = self.build_nn_model(input_dim=X_train_scaled.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train_scaled, y_train,
                            validation_data=(X_val_scaled, y_val),
                            epochs=100, batch_size=32,
                            callbacks=[early_stop], verbose=2)
        train_loss, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
        val_loss, val_mae = model.evaluate(X_val_scaled, y_val, verbose=0)
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_r2": train_r2,
            "val_r2": val_r2
        }
        elapsed = time.time() - start_time
        print("[NN SELECTED Features] Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        plt.figure(figsize=(8,6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Neural Network SELECTED Features Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.savefig("nn_selected_loss_curve.png")
        plt.close()
        print(f"Neural Network SELECTED features training completed in {elapsed:.2f} seconds.")
        self.save_metrics(metrics, "metrics_nn_selected_features.xlsx")
        self.plot_learning_curve(model, X_train_scaled, y_train, title="NN Selected Features Learning Curve", cv=3,
                                 filename_prefix="nn_selected_learning_curve")
        return metrics, model

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
                     [metrics_dict.get("train_mse", metrics_dict.get("train_loss")),
                      metrics_dict.get("val_mse", metrics_dict.get("val_loss"))],
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

    def plot_nn_learning_curve_manual(self, build_fn, X, y, filename_prefix="nn_learning_curve_manual"):
        """
        Manually plot a learning curve for a Keras NN by training on increasing subset sizes.
        """
        from sklearn.model_selection import KFold
        train_sizes = np.linspace(0.1, 1.0, 5)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores_train = []
        scores_val = []

        for frac in train_sizes:
            train_r2s = []
            val_r2s = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Only use a fraction of training data
                subset_size = int(frac * len(X_train))
                X_train_subset = X_train[:subset_size]
                y_train_subset = y_train[:subset_size]

                model = build_fn(X.shape[1])
                model.fit(X_train_subset, y_train_subset, epochs=10, batch_size=32, verbose=0)

                y_train_pred = model.predict(X_train_subset)
                y_val_pred = model.predict(X_val)

                train_r2s.append(r2_score(y_train_subset, y_train_pred))
                val_r2s.append(r2_score(y_val, y_val_pred))

            scores_train.append(np.mean(train_r2s))
            scores_val.append(np.mean(val_r2s))
            print(f"✅ Done with fraction: {frac:.1f}")

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, scores_train, 'o-', label='Training R²')
        plt.plot(train_sizes, scores_val, 'o-', label='Validation R²')
        plt.title("Manual NN Learning Curve")
        plt.xlabel("Training Set Fraction")
        plt.ylabel("R² Score")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        filename = f"{filename_prefix}.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ Saved manual NN learning curve to: {filename}")

    def plot_learning_curve(self, estimator, X, y, title="Learning Curve", cv=None,
                            n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5),
                            filename_prefix="learning_curve", build_fn=None):
        """
        Plot a learning curve safely for any estimator (including Keras).
        """
        if cv is None:
            cv = KFold(n_splits=3, shuffle=True, random_state=42)  # 👈 Avoids is_classifier()

        # If it's not a proper sklearn estimator, wrap it with KerasRegressor
        if not isinstance(estimator, BaseEstimator):
            if build_fn is None:
                raise ValueError("You must provide `build_fn` if passing a raw Keras model.")
            input_dim = X.shape[1]
            estimator = KerasRegressor(
                model=build_fn,
                model__input_dim=input_dim,
                epochs=100,
                batch_size=32,
                verbose=0
            )

        # Generate learning curve data
        train_sizes, train_scores, cv_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs,
            train_sizes=train_sizes, scoring='r2'
        )

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="red", label="Training score")
        plt.plot(train_sizes, np.mean(cv_scores, axis=1), 'o-', color="green", label="Cross-validation score")
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
        Run training for Random Forest, GLM, XGBoost, and Neural Network models on all features and selected features.
        Save evaluation metrics and plots to Excel/PNG, plot learning curves, and print all metrics.
        """
        # ------------------ Random Forest Training ------------------ #
        print("\n=== Starting Random Forest Training ===")
        # Uncomment if you want to run RF training.
        # rf_metrics_all, rf_model_all = self.train_rf_all_features()
        # self.save_metrics(rf_metrics_all, "metrics_rf_all_features.xlsx")
        # self.plot_metrics(rf_metrics_all, "loss_rf_all_features")
        # X_all = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        # y_all = self.df_encoded["target"]
        # self.plot_learning_curve(rf_model_all, X_all, y_all, title="RF All Features Learning Curve", cv=3,
        #                          filename_prefix="rf_all_learning_curve")
        #
        # rf_metrics_sel, rf_model_sel = self.train_rf_selected_features()
        # self.save_metrics(rf_metrics_sel, "metrics_rf_selected_features.xlsx")
        # self.plot_metrics(rf_metrics_sel, "loss_rf_selected_features")
        # X_sel = self.df_encoded[self.selected_features]
        # self.plot_learning_curve(rf_model_sel, X_sel, y_all, title="RF Selected Features Learning Curve", cv=3,
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
        xgb_metrics_all, xgb_model_all = self.train_xgb_all_features()
        self.save_metrics(xgb_metrics_all, "metrics_xgb_all_features.xlsx")
        self.plot_metrics(xgb_metrics_all, "loss_xgb_all_features")
        X_all = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y_all = self.df_encoded["target"]
        self.plot_learning_curve(xgb_model_all, X_all, y_all, title="XGB All Features Learning Curve", cv=3,
                                 filename_prefix="xgb_all_learning_curve")
        xgb_metrics_sel, xgb_model_sel = self.train_xgb_selected_features()
        self.save_metrics(xgb_metrics_sel, "metrics_xgb_selected_features.xlsx")
        self.plot_metrics(xgb_metrics_sel, "loss_xgb_selected_features")
        X_sel = self.df_encoded[self.selected_features]
        self.plot_learning_curve(xgb_model_sel, X_sel, y_all, title="XGB Selected Features Learning Curve", cv=3,
                                 filename_prefix="xgb_selected_learning_curve")

        # ------------------ Neural Network Training ------------------ #
        print("\n=== Starting Neural Network Training ===")
        nn_metrics_all, nn_model_all = self.train_nn_all_features()
        self.save_metrics(nn_metrics_all, "metrics_nn_all_features.xlsx")
        self.plot_metrics(nn_metrics_all, "loss_nn_all_features")
        X_all = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y_all = self.df_encoded["target"]
        self.plot_learning_curve(nn_model_all, StandardScaler().fit_transform(X_all), y_all,
                                 title="NN All Features Learning Curve", cv=3,
                                 filename_prefix="nn_all_learning_curve")
        nn_metrics_sel, nn_model_sel = self.train_nn_selected_features()
        self.save_metrics(nn_metrics_sel, "metrics_nn_selected_features.xlsx")
        self.plot_metrics(nn_metrics_sel, "loss_nn_selected_features")
        X_sel = self.df_encoded[self.selected_features]
        self.plot_learning_curve(nn_model_sel, StandardScaler().fit_transform(X_sel), y_all,
                                 title="NN Selected Features Learning Curve", cv=3,
                                 filename_prefix="nn_selected_learning_curve")
        print("\n=== All Model Training Completed ===")

    # ------------------ Neural Network Methods ------------------ #

    def train_nn_all_features(self):
        print("Starting Neural Network training on ALL features...")
        start_time = time.time()
        X = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model = self.build_nn_model(input_dim=X_train_scaled.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train_scaled, y_train,
                            validation_data=(X_val_scaled, y_val),
                            epochs=100, batch_size=32,
                            callbacks=[early_stop], verbose=2)
        train_loss, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
        val_loss, val_mae = model.evaluate(X_val_scaled, y_val, verbose=0)
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_r2": train_r2,
            "val_r2": val_r2
        }
        elapsed = time.time() - start_time
        print("[NN ALL Features] Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        plt.figure(figsize=(8,6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Neural Network ALL Features Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.savefig("nn_all_loss_curve.png")
        plt.close()
        print(f"Neural Network ALL features training completed in {elapsed:.2f} seconds.")
        return metrics, model

    def train_nn_selected_features(self):
        if not self.selected_features:
            raise ValueError("No selected features available from PCA.")
        print("Starting Neural Network training on SELECTED features...")
        start_time = time.time()
        X = self.df_encoded[self.selected_features]
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model = self.build_nn_model(input_dim=X_train_scaled.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train_scaled, y_train,
                            validation_data=(X_val_scaled, y_val),
                            epochs=100, batch_size=32,
                            callbacks=[early_stop], verbose=2)
        train_loss, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
        val_loss, val_mae = model.evaluate(X_val_scaled, y_val, verbose=0)
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_r2": train_r2,
            "val_r2": val_r2
        }
        elapsed = time.time() - start_time
        print("[NN SELECTED Features] Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        plt.figure(figsize=(8,6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Neural Network SELECTED Features Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.savefig("nn_selected_loss_curve.png")
        plt.close()
        print(f"Neural Network SELECTED features training completed in {elapsed:.2f} seconds.")
        return metrics, model

    # ------------------ Utility Methods ------------------ #
    @staticmethod
    def save_metrics(metrics_dict: dict, filename: str):
        df_metrics = pd.DataFrame(metrics_dict, index=[0])
        df_metrics.to_excel(filename, index=False)
        print(f"✅ Saved metrics to: {filename}")

    @staticmethod
    def plot_metrics(metrics_dict: dict, filename_prefix: str):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(["Train MSE", "Validation MSE"],
                     [metrics_dict.get("train_mse", metrics_dict.get("train_loss")),
                      metrics_dict.get("val_mse", metrics_dict.get("val_loss"))],
                     marker="o", linestyle="--", color="blue")
        axes[0].set_title("Mean Squared Error")
        axes[0].set_ylabel("MSE")
        axes[1].plot(["Train MAE", "Validation MAE"],
                     [metrics_dict["train_mae"], metrics_dict["val_mae"]],
                     marker="o", linestyle="--", color="orange")
        axes[1].set_title("Mean Absolute Error")
        axes[2].plot(["Train R²", "Validation R²"],
                     [metrics_dict["train_r2"], metrics_dict["val_r2"]],
                     marker="o", linestyle="--", color="green")
        axes[2].set_title("R² Score")
        fig.tight_layout()
        filename = f"{filename_prefix}_metrics.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ Saved metrics plot to: {filename}")



    def run_training(self):
        #print("\n=== Starting XGBoost Training ===")
        #xgb_metrics_all, xgb_model_all = self.train_xgb_all_features()
        #self.save_metrics(xgb_metrics_all, "metrics_xgb_all_features.xlsx")
        #self.plot_metrics(xgb_metrics_all, "loss_xgb_all_features")
        #X_all = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        #y_all = self.df_encoded["target"]
        #self.plot_learning_curve(xgb_model_all, X_all, y_all, title="XGB All Features Learning Curve", cv=3,
        #                         filename_prefix="xgb_all_learning_curve")
        #xgb_metrics_sel, xgb_model_sel = self.train_xgb_selected_features()
        #self.save_metrics(xgb_metrics_sel, "metrics_xgb_selected_features.xlsx")
        #self.plot_metrics(xgb_metrics_sel, "loss_xgb_selected_features")
        #X_sel = self.df_encoded[self.selected_features]
        #self.plot_learning_curve(xgb_model_sel, X_sel, y_all, title="XGB Selected Features Learning Curve", cv=3,
         #                        filename_prefix="xgb_selected_learning_curve")

        print("\n=== Starting Neural Network Training ===")
        nn_metrics_all, nn_model_all = self.train_nn_all_features()
        self.save_metrics(nn_metrics_all, "metrics_nn_all_features.xlsx")
        self.plot_metrics(nn_metrics_all, "loss_nn_all_features")
        X_all = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y_all = self.df_encoded["target"]
        #self.plot_learning_curve(nn_model_all, StandardScaler().fit_transform(X_all), y_all, title="NN All Features Learning Curve", cv=3,
        #                         filename_prefix="nn_all_learning_curve")
        nn_metrics_sel, nn_model_sel = self.train_nn_selected_features()
        self.save_metrics(nn_metrics_sel, "metrics_nn_selected_features.xlsx")
        self.plot_metrics(nn_metrics_sel, "loss_nn_selected_features")
        X_sel = self.df_encoded[self.selected_features]
        self.plot_learning_curve(nn_model_sel, StandardScaler().fit_transform(X_sel), y_all, title="NN Selected Features Learning Curve", cv=3,
                                 filename_prefix="nn_selected_learning_curve")

        print("\n=== All Model Training Completed ===")

    def train_xgb_selected_features(self):
        if not self.selected_features:
            raise ValueError("No selected features available from PCA.")
        print("Starting XGBoost training on SELECTED features with early stopping...")
        start_time = time.time()
        X = self.df_encoded[self.selected_features]
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4],
            'learning_rate': [0.1, 0.05],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'random_state': [self.random_state]
        }
        xgb_model = XGBRegressor(objective='reg:squarederror', verbosity=1)
        grid = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error',
                            n_jobs=-1, verbose=2)
        grid.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 early_stopping_rounds=10,
                 verbose=False)
        best_model = grid.best_estimator_
        cv_results_df = pd.DataFrame(grid.cv_results_)
        cv_results_df.to_excel("cv_results_xgb_selected_features.xlsx", index=False)
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
        print("[XGB SELECTED Features] Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print(f"XGBoost SELECTED features training completed in {elapsed:.2f} seconds.")
        return metrics, best_model



    def build_nn_model_selected(self, input_dim):
        # Improved NN using an explicit Input layer, deeper architecture, and a lower learning rate.
        inputs = Input(shape=(input_dim,))
        x = Dense(128, activation='relu')(inputs)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)

        outputs = Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
        return model

    def train_nn_all_features(self):
        print("Starting Neural Network training on ALL features...")
        start_time = time.time()
        X = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model = self.build_nn_model(input_dim=X_train_scaled.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train_scaled, y_train,
                            validation_data=(X_val_scaled, y_val),
                            epochs=100, batch_size=32,
                            callbacks=[early_stop], verbose=2)
        train_loss, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
        val_loss, val_mae = model.evaluate(X_val_scaled, y_val, verbose=0)
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_r2": train_r2,
            "val_r2": val_r2
        }
        elapsed = time.time() - start_time
        print("[NN ALL Features] Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        plt.figure(figsize=(8,6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Neural Network ALL Features Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.savefig("nn_all_loss_curve.png")
        plt.close()
        print(f"Neural Network ALL features training completed in {elapsed:.2f} seconds.")
        return metrics, model

    def train_nn_selected_features(self):
        if not self.selected_features:
            raise ValueError("No selected features available from PCA.")
        print("Starting Neural Network training on SELECTED features...")
        start_time = time.time()
        X = self.df_encoded[self.selected_features]
        y = self.df_encoded["target"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model = self.build_nn_model_selected(input_dim=X_train_scaled.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train_scaled, y_train,
                            validation_data=(X_val_scaled, y_val),
                            epochs=100, batch_size=32,
                            callbacks=[early_stop], verbose=2)
        train_loss, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
        val_loss, val_mae = model.evaluate(X_val_scaled, y_val, verbose=0)
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_r2": train_r2,
            "val_r2": val_r2
        }
        elapsed = time.time() - start_time
        print("[NN SELECTED Features] Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        plt.figure(figsize=(8,6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Neural Network SELECTED Features Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)
        plt.savefig("nn_selected_loss_curve.png")
        plt.close()
        print(f"Neural Network SELECTED features training completed in {elapsed:.2f} seconds.")
        return metrics, model

    # ------------------ Utility Methods ------------------ #
    @staticmethod
    def save_metrics(metrics_dict: dict, filename: str):
        df_metrics = pd.DataFrame(metrics_dict, index=[0])
        df_metrics.to_excel(filename, index=False)
        print(f"✅ Saved metrics to: {filename}")

    @staticmethod
    def plot_metrics(metrics_dict: dict, filename_prefix: str):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(["Train MSE", "Validation MSE"],
                     [metrics_dict.get("train_mse", metrics_dict.get("train_loss")),
                      metrics_dict.get("val_mse", metrics_dict.get("val_loss"))],
                     marker="o", linestyle="--", color="blue")
        axes[0].set_title("Mean Squared Error")
        axes[0].set_ylabel("MSE")
        axes[1].plot(["Train MAE", "Validation MAE"],
                     [metrics_dict["train_mae"], metrics_dict["val_mae"]],
                     marker="o", linestyle="--", color="orange")
        axes[1].set_title("Mean Absolute Error")
        axes[2].plot(["Train R²", "Validation R²"],
                     [metrics_dict["train_r2"], metrics_dict["val_r2"]],
                     marker="o", linestyle="--", color="green")
        axes[2].set_title("R² Score")
        fig.tight_layout()
        filename = f"{filename_prefix}_metrics.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ Saved metrics plot to: {filename}")



    def run_training(self):

        # ------------------ Random Forest Training ------------------ #
        # Uncomment if you want to run RF training.
        #print("\n=== Starting Random Forest Training ===")
        #rf_metrics_all, rf_model_all = self.train_rf_all_features()
        #self.save_metrics(rf_metrics_all, "metrics_rf_all_features.xlsx")
        #self.plot_metrics(rf_metrics_all, "loss_rf_all_features")
        #X_all = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        #y_all = self.df_encoded["target"]
        #self.plot_learning_curve(rf_model_all, X_all, y_all, title="RF All Features Learning Curve", cv=3,
        #                         filename_prefix="rf_all_learning_curve")

        """
        print("\n=== Starting XGBoost Training ===")
        xgb_metrics_all, xgb_model_all = self.train_xgb_all_features()
        self.save_metrics(xgb_metrics_all, "metrics_xgb_all_features.xlsx")
        self.plot_metrics(xgb_metrics_all, "loss_xgb_all_features")
        X_all = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y_all = self.df_encoded["target"]
        self.plot_learning_curve(xgb_model_all, X_all, y_all, title="XGB All Features Learning Curve", cv=3,
                                 filename_prefix="xgb_all_learning_curve")
        xgb_metrics_sel, xgb_model_sel = self.train_xgb_selected_features()
        self.save_metrics(xgb_metrics_sel, "metrics_xgb_selected_features.xlsx")
        self.plot_metrics(xgb_metrics_sel, "loss_xgb_selected_features")
        X_sel = self.df_encoded[self.selected_features]
        self.plot_learning_curve(xgb_model_sel, X_sel, y_all, title="XGB Selected Features Learning Curve", cv=3,
                                 filename_prefix="xgb_selected_learning_curve")
        """

        print("\n=== Starting Neural Network Training ===")
        nn_metrics_all, nn_model_all = self.train_nn_all_features()
        self.save_metrics(nn_metrics_all, "metrics_nn_all_features.xlsx")
        self.plot_metrics(nn_metrics_all, "loss_nn_all_features")
        X_all = self.df_encoded.drop(columns=["IDpol", "target"], errors='ignore')
        y_all = self.df_encoded["target"]
        X_all_scaled = StandardScaler().fit_transform(X_all)
        y_all_array = np.array(y_all)

        self.plot_nn_learning_curve_manual(
            build_fn=self.build_nn_model,
            X=X_all_scaled,
            y=y_all_array,
            filename_prefix="nn_all_learning_curve_manual"
        )
        """
        nn_metrics_sel, nn_model_sel = self.train_nn_selected_features()
        self.save_metrics(nn_metrics_sel, "metrics_nn_selected_features.xlsx")
        self.plot_metrics(nn_metrics_sel, "loss_nn_selected_features")
        X_sel = self.df_encoded[self.selected_features]
        X_sel_scaled = StandardScaler().fit_transform(X_sel)
        self.plot_nn_learning_curve_manual(
            build_fn=self.build_nn_model_selected,
            X=X_sel_scaled,
            y=y_all_array,
            filename_prefix="nn_selected_learning_curve_manual"
        )
        """

        print("\n=== All Model Training Completed ===")


# ---------------------- USAGE EXAMPLE ---------------------- #
if __name__ == "__main__":

    print("Running..........................")
    # Define column lists.
    freq_cols = [
        "IDpol", "ClaimNb", "Exposure", "Area", "VehPower",
        "VehAge", "DrivAge", "BonusMalus", "VehBrand",
        "VehGas", "Density", "Region"
    ]
    sev_cols = ["IDpol", "ClaimAmount"]
    categorical_cols = ["Area", "VehBrand", "VehGas", "Region"]
    exclude_from_pca = ["IDpol", "target", "ClaimNb", "Exposure", "total_claim_amount"]

    # Initialize the preprocessor.
    preprocessor = InsurancePreprocessor(
        freq_filename="Data/freMTPL2freq.arff",
        sev_filename="Data/freMTPL2sev.arff",
        freq_cols=freq_cols,
        sev_cols=sev_cols,
        categorical_cols=categorical_cols,
        exclude_from_pca=exclude_from_pca,
        pca_n_components=5
    )

    # Initialize the model trainer with the preprocessor.
    trainer = InsuranceModelTrainer(preprocessor=preprocessor, test_size=0.2, random_state=42)

    # Run training for all models (RF, GLM, XGBoost, and Neural Network on both all and selected features).
    trainer.run_training()
