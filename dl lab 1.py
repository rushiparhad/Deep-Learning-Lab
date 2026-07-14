import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
DATASET_PATH = "Housing.csv"
TARGET_COLUMN = "price"


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_preprocess_data(dataset_path=DATASET_PATH):
    data = pd.read_csv(dataset_path)

    x = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype("float32")

    numeric_features = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [column for column in x.columns if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED
    )

    x_train = preprocessor.fit_transform(x_train).astype("float32")
    x_test = preprocessor.transform(x_test).astype("float32")

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).reshape(-1)

    return (
        x_train,
        x_test,
        y_train.to_numpy(),
        y_train_scaled.astype("float32"),
        y_test.to_numpy(),
        preprocessor,
        target_scaler,
    )


def build_dffnn(input_dim, hidden_layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))

    for neurons in hidden_layers:
        model.add(tf.keras.layers.Dense(neurons, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.10))

    model.add(tf.keras.layers.Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return model


def train_and_evaluate_architecture(
    x_train, x_test, y_train_scaled, y_test, target_scaler, hidden_layers
):
    model = build_dffnn(input_dim=x_train.shape[1], hidden_layers=hidden_layers)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train_scaled,
        validation_split=0.2,
        epochs=300,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0,
    )

    scaled_predictions = model.predict(x_test, verbose=0).reshape(-1, 1)
    predictions = target_scaler.inverse_transform(scaled_predictions).reshape(-1)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    return {
        "layers": len(hidden_layers),
        "architecture": str(hidden_layers),
        "epochs": len(history.history["loss"]),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }


def main():
    set_seed()

    x_train, x_test, _, y_train_scaled, y_test, _, target_scaler = load_and_preprocess_data()

    architectures = [
        [64],
        [128, 64],
        [128, 64, 32],
        [256, 128, 64, 32],
        [256, 128, 64, 32, 16],
    ]

    results = []
    for hidden_layers in architectures:
        print(f"Training DFFNN architecture: {hidden_layers}", flush=True)
        result = train_and_evaluate_architecture(
            x_train, x_test, y_train_scaled, y_test, target_scaler, hidden_layers
        )
        results.append(result)

    results_df = pd.DataFrame(results).sort_values(by="RMSE")

    print("\nDFFNN House Price Prediction Results")
    print(results_df.to_string(index=False))

    best_model = results_df.iloc[0]
    print("\nBest architecture based on RMSE:")
    print(
        f"{best_model['architecture']} | MAE={best_model['MAE']:.2f} | "
        f"MSE={best_model['MSE']:.2f} | RMSE={best_model['RMSE']:.2f}"
    )


if __name__ == "__main__":
    main()
