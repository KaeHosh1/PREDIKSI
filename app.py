from flask import Flask, request, render_template
import joblib
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import io, base64

app = Flask(__name__)

# Load model yang sudah dilatih
model = joblib.load("prediksi_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Ambil input dari form
        hours_studied = float(request.form["hours_studied"])
        previous_scores = float(request.form["previous_scores"])
        sleep_hours = float(request.form["sleep_hours"])
        sample_papers = float(request.form["sample_papers"])
        extracurricular = int(request.form["extracurricular"])

        # Buat array input
        X = np.array([[hours_studied, previous_scores, sleep_hours, sample_papers, extracurricular]])

        # Prediksi performance index
        y_pred = model.predict(X)[0]

        # Buat grafik
        plt.figure(figsize=(5,3))
        plt.bar(["Prediksi Performance Index"], [y_pred], color="skyblue")
        plt.ylim(0, 100)
        plt.title("Prediksi Kinerja Siswa")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        # Kesimpulan sederhana
        if y_pred >= 80:
            kesimpulan = "Kinerja kamu sangat baik! Pertahankan terus."
            solusi = "Cobalah membantu teman belajar untuk mempertajam pemahamanmu."
        elif y_pred >= 60:
            kesimpulan = "Kinerja cukup baik, tapi masih bisa ditingkatkan."
            solusi = "Perbaiki manajemen waktu belajar & tidur."
        else:
            kesimpulan = "Kinerja masih kurang optimal."
            solusi = "Tambah jam belajar, perbanyak latihan soal, dan tidur yang cukup."

        return render_template("index.html",
                                prediction=y_pred,
                                img_data=img_base64,
                                kesimpulan=kesimpulan,
                                solusi=solusi)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)