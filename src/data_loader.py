"""Dieses Modul enthält Funktionen zum Laden, Normalisieren und Vorbereiten der N-CMAPSS-Daten"""

# pylint: disable=import-error
import h5py
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# pylint: disable=invalid-name
# Variablennamen entsprechen den Spaltennamen im Datensatz, um Verwirrung zu vermeiden

# pylint: disable=too-many-locals
# notwendig, um alle Variablen zu laden und in DataFrames umzuwandeln

def load_data(filename: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Lädt den N-CMAPSS Datensatz aus einer HDF5-Datei und gibt Trainings- und Testdaten zurück

    Args:
        filename (str): Pfad zur HDF5-Datei (.h5) mit dem N-CMAPSS Datensatz

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list[str]]:
            - df_training: Trainingsdaten als DataFrame
            - df_testing: Testdaten als DataFrame
            - new_col_names: Spaltennamen nach dem Entfernen nicht relevanter Features
    """
    # Load data
    print("Loading data...")

    try:
        with h5py.File(filename, "r") as hdf:
            # Development set
            W_dev: np.ndarray = np.array(hdf.get("W_dev"))  # W
            X_s_dev: np.ndarray = np.array(hdf.get("X_s_dev"))  # X_s
            X_v_dev: np.ndarray = np.array(hdf.get("X_v_dev"))  # X_v
            T_dev: np.ndarray = np.array(hdf.get("T_dev"))  # T
            Y_dev: np.ndarray = np.array(hdf.get("Y_dev"))  # RUL
            A_dev: np.ndarray = np.array(hdf.get("A_dev"))  # Auxiliary

            # Test set
            W_test: np.ndarray = np.array(hdf.get("W_test"))  # W
            X_s_test: np.ndarray = np.array(hdf.get("X_s_test"))  # X_s
            X_v_test: np.ndarray = np.array(hdf.get("X_v_test"))  # X_v
            T_test: np.ndarray = np.array(hdf.get("T_test"))  # T
            Y_test: np.ndarray = np.array(hdf.get("Y_test"))  # RUL
            A_test: np.ndarray = np.array(hdf.get("A_test"))  # Auxiliary

            # Column names
            W_var: np.ndarray = np.array(hdf.get("W_var"))
            X_s_var: np.ndarray = np.array(hdf.get("X_s_var"))
            X_v_var: np.ndarray = np.array(hdf.get("X_v_var"))
            T_var: np.ndarray = np.array(hdf.get("T_var"))
            A_var: np.ndarray = np.array(hdf.get("A_var"))

            # from np.array to list dtype U4/U5
            W_var: list[str] = list(np.array(W_var, dtype="U20"))
            X_s_var: list[str] = list(np.array(X_s_var, dtype="U20"))
            X_v_var: list[str] = list(np.array(X_v_var, dtype="U20"))
            T_var: list[str] = list(np.array(T_var, dtype="U20"))
            A_var: list[str] = list(np.array(A_var, dtype="U20"))

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        raise

    # Combine features and create DataFrames
    col_names: np.ndarray = np.concatenate(
        (A_var, W_var, X_s_var, X_v_var, T_var, ["RUL"])
    )
    X_train: np.ndarray = np.concatenate(
        (A_dev, W_dev, X_s_dev, X_v_dev, T_dev), axis=1
    )
    Y_train: np.ndarray = Y_dev
    training_set: np.ndarray = np.concatenate((X_train, Y_train), axis=1)
    df_training: pd.DataFrame = pd.DataFrame(
        data=training_set, columns=col_names
    )

    X_test: np.ndarray = np.concatenate(
        (A_test, W_test, X_s_test, X_v_test, T_test), axis=1
    )
    Y_test: np.ndarray = Y_test
    testing_set: np.ndarray = np.concatenate((X_test, Y_test), axis=1)
    df_testing: pd.DataFrame = pd.DataFrame(
        data=testing_set, columns=col_names
    )

    # drop irrelevant columns, based on previous data analysis
    try:
        df_training: pd.DataFrame = df_training.drop(
            columns=[
                "LPT_flow_mod",
                "LPT_eff_mod",
                "HPT_flow_mod",
                "HPC_flow_mod",
                "HPC_eff_mod",
                "LPC_flow_mod",
                "LPC_eff_mod",
                "fan_flow_mod",
                "fan_eff_mod",
            ]
        )
        df_testing: pd.DataFrame = df_testing.drop(
            columns=[
                "LPT_flow_mod",
                "LPT_eff_mod",
                "HPT_flow_mod",
                "HPC_flow_mod",
                "HPC_eff_mod",
                "LPC_flow_mod",
                "LPC_eff_mod",
                "fan_flow_mod",
                "fan_eff_mod",
            ]
        )
    except KeyError as e:
        print(
            f"Warning: Some columns were not \
              found and could not be dropped: {e}"
        )

    new_col_names: pd.Index = df_training.columns

    print("data loaded")

    return df_training, df_testing, new_col_names


def create_testing_and_training_sets(
    df_training_scaled: pd.DataFrame,
    df_testing_scaled: pd.DataFrame,
    win_len=10,
    batch_size=64,
) -> tuple[DataLoader, DataLoader]:
    """
    Erzeugt Sliding-Window-Datensätze und gibt DataLoader für Training und Test zurück

    Für jede Triebwerkseinheit werden alle möglichen Fenster der Länge win_len
    extrahiert. Der RUL-Wert des letzten Zyklus im Fenster dient als Zielgröße.

    Args:
        df_training_scaled (pd.DataFrame): Normalisierte Trainingsdaten
        df_testing_scaled (pd.DataFrame): Normalisierte Testdaten
        win_len (int): Fensterlänge in Zyklen. Standard: 10
        batch_size (int): Batch-Größe für den DataLoader. Default: 256

    Returns:
        tuple[DataLoader, DataLoader]: DataLoader für Trainings- und Testdaten
    """

    print("Create training and testing dataset...")

    # ===== TRAINING DATA =====
    X_train_model: list[np.ndarray] = []
    y_train_model: list[float] = []

    for unit in tqdm.tqdm(df_training_scaled["unit"].unique(), leave=False):
        unit_data: pd.DataFrame = df_training_scaled[
            df_training_scaled["unit"] == unit
        ]

        for i in range(0, len(unit_data) - win_len + 1):
            temp: pd.DataFrame = unit_data.iloc[i : i + win_len]
            x_temp: pd.DataFrame = temp.drop(columns=["unit", "RUL"])
            y_temp: float = temp["RUL"].values[-1]
            X_train_model.append(x_temp.values)
            y_train_model.append(y_temp)

    # create torch tensors
    X_train_t: torch.Tensor = torch.tensor(X_train_model, dtype=torch.float32)
    y_train_t: torch.Tensor = torch.tensor(y_train_model, dtype=torch.float32)

    print(f" X_train shape: {X_train_t.shape}")

    # create Torch DataLoader
    train_dataset: TensorDataset = TensorDataset(X_train_t, y_train_t)
    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # ===== TESTING DATA =====
    X_test_model: list[np.ndarray] = []
    y_test_model: list[float] = []

    for unit in tqdm.tqdm(df_testing_scaled["unit"].unique(), leave=False):
        unit_data: pd.DataFrame = df_testing_scaled[
            df_testing_scaled["unit"] == unit
        ]

        for i in range(0, len(unit_data) - win_len + 1):
            temp: pd.DataFrame = unit_data.iloc[i : i + win_len]
            x_temp: pd.DataFrame = temp.drop(columns=["unit", "RUL"])
            y_temp: float = temp["RUL"].values[-1]
            X_test_model.append(x_temp.values)
            y_test_model.append(y_temp)

    # create torch tensors
    X_test_t: torch.Tensor = torch.tensor(X_test_model, dtype=torch.float32)
    y_test_t: torch.Tensor = torch.tensor(y_test_model, dtype=torch.float32)

    print(f" X_test shape: {X_test_t.shape}")

    # create torch TestLoader
    test_dataset: TensorDataset = TensorDataset(X_test_t, y_test_t)
    test_loader: DataLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    print("Datasets created!")

    return (
        train_loader,
        test_loader,
    )


def normalize_data(
    df_training: pd.DataFrame,
    df_testing: pd.DataFrame,
    new_col_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalisiert Trainings- und Testdaten mit dem StandardScaler

    Der Scaler wird ausschließlich auf den Trainingsdaten gefittet und
    anschließend auf die Testdaten angewendet, um Datenlecks zu vermeiden.
    Einheit (unit) und Zielgröße (RUL) werden nicht skaliert.

    Args:
        df_training (pd.DataFrame): Rohe Trainingsdaten
        df_testing (pd.DataFrame): Rohe Testdaten
        new_col_names (list[str]): Spaltennamen für die skalierten DataFrames

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Normalisierte Trainings- und Testdaten
    """

    print("Normalize Data...")
    scaler: StandardScaler = StandardScaler()
    features: pd.DataFrame = df_training.drop(columns=["RUL", "unit"])
    X_train: np.ndarray = scaler.fit_transform(features)

    features_testing: pd.DataFrame = df_testing.drop(columns=["RUL", "unit"])
    X_test: np.ndarray = scaler.transform(features_testing)

    df_training_scaled: pd.DataFrame = pd.DataFrame(
        data=(
            np.concatenate(
                (
                    np.array(df_training[["unit"]]),
                    X_train,
                    np.array(df_training[["RUL"]]),
                ),
                axis=1,
            )
        ),
        columns=new_col_names,
    )

    df_testing_scaled: pd.DataFrame = pd.DataFrame(
        data=(
            np.concatenate(
                (
                    np.array(df_testing[["unit"]]),
                    X_test,
                    np.array(df_testing[["RUL"]]),
                ),
                axis=1,
            )
        ),
        columns=new_col_names,
    )

    print("Data Normalized")
    return df_training_scaled, df_testing_scaled
