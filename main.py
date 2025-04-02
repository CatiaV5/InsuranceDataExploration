from preprocessor import InsurancePreprocessor
from Archive.training import InsuranceModelTrainer


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
