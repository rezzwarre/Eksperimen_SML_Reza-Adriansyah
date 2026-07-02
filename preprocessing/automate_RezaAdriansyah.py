"""
automate_RezaAdriansyah.py

Script ini berisi fungsi-fungsi untuk melakukan preprocessing dataset
"extended_flower_morphometrics.csv" secara otomatis, mengikuti tahapan
yang sama dengan proses eksperimen manual pada
Eksperimen_RezaAdriansyah.ipynb, yaitu:

1. Load data
2. Menangani missing values
3. Menangani data duplikat
4. Standarisasi fitur numerik (StandardScaler)
5. Penanganan outlier menggunakan metode IQR
6. Encoding variabel kategorikal (LabelEncoder)
7. Binning fitur native_altitude_m

Fungsi utama `preprocess_data()` mengembalikan dataset yang sudah siap
dilatih (bersih, ternormalisasi, dan terenkode) sekaligus menyimpannya
ke file CSV baru.
"""

import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(input_path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari path CSV yang diberikan."""
    df = pd.read_csv(input_path)
    return df


def handle_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris dengan missing value dan baris duplikat."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def scale_numerical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, list]:
    """Melakukan standarisasi (StandardScaler) pada seluruh kolom numerik."""
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler, numerical_cols


def remove_outliers_iqr(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """Menghapus outlier pada kolom numerik menggunakan metode IQR."""
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def encode_categorical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Melakukan Label Encoding pada seluruh kolom kategorikal (object)."""
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for column in categorical_cols:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        encoders[column] = label_encoder
    return df, encoders


def bin_altitude_feature(df: pd.DataFrame,
                          column: str = "native_altitude_m",
                          new_column: str = "altitude_group") -> pd.DataFrame:
    """Melakukan binning pada kolom native_altitude_m menjadi 3 kelompok."""
    if column in df.columns:
        df[new_column] = pd.qcut(df[column], q=3, labels=["Low", "Medium", "High"])
    return df


def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing secara otomatis dan
    mengembalikan dataframe yang siap digunakan untuk training model.

    Parameters
    ----------
    input_path : str
        Path menuju dataset mentah (raw).
    output_path : str
        Path untuk menyimpan hasil dataset yang sudah diproses.

    Returns
    -------
    pd.DataFrame
        Dataset hasil preprocessing.
    """
    df = load_data(input_path)
    df = handle_missing_and_duplicates(df)
    df, scaler, numerical_cols = scale_numerical_features(df)
    df = remove_outliers_iqr(df, numerical_cols)
    df, encoders = encode_categorical_features(df)
    df = bin_altitude_feature(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessing selesai. Dataset tersimpan di: {output_path}")
    print(f"Jumlah baris akhir: {df.shape[0]}, jumlah kolom: {df.shape[1]}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automasi preprocessing dataset flower morphometrics.")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("..", "namadataset_raw", "extended_flower_morphometrics.csv"),
        help="Path menuju dataset mentah (raw CSV).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("namadataset_preprocessing", "flower_morphometrics_preprocessed.csv"),
        help="Path untuk menyimpan dataset hasil preprocessing.",
    )
    args = parser.parse_args()

    preprocess_data(args.input, args.output)
