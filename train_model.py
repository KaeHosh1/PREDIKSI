import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# ==========================
# 1. Baca dataset
# ==========================
df = pd.read_csv("Student_Performance.csv")

# Cek beberapa data pertama (opsional, bisa dihapus kalau mau)
print("Preview Dataset:")
print(df.head())

# ==========================
# 2. Preprocessing
# ==========================

# Ubah kolom Extracurricular Activities menjadi kategori numerik (Yes=1, No=0)
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})


# ==========================
# 3. Siapkan fitur (X) dan target (y)
# ==========================
X = df[["Hours Studied", "Previous Scores", "Sleep Hours", "Sample Question Papers Practiced", "Extracurricular Activities"]]
y = df["Performance Index"]

# ==========================
# 4. Buat & latih model Linear Regression
# ==========================
model = LinearRegression()
model.fit(X, y)

# ==========================
# 5. Simpan model ke file prediksi_model.pkl
# ==========================
joblib.dump(model, "prediksi_model.pkl")

print("✅ Model berhasil dilatih dan disimpan sebagai prediksi_model.pkl")