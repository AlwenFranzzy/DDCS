# model.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import numpy as np

def train_mlp(df, random_state=1):
    """
    Melatih model MLPRegressor untuk memprediksi next_recycled
    berdasarkan fitur:
    [pdam_prev, recycled_before, hour, day, action, demand]

    Parameter:
    - df : DataFrame hasil koleksi data (data.py)
    - random_state : untuk memastikan hasil training konsisten

    Output:
    - model  : model MLP terlatih
    - scaler : normalizer fitur (StandardScaler)
    - score  : skor R^2 pada data test untuk evaluasi performa model
    """

    # ----------------------------------------------------------------------
    # 1. Ekstraksi fitur input (X) dan target output (y)
    # ----------------------------------------------------------------------
    X = df[['pdam_prev', 'recycled_before', 'hour', 'day', 'action', 'demand']].values
    y = df['next_recycled'].values

    # ----------------------------------------------------------------------
    # 2. Normalisasi fitur dengan StandardScaler
    #    - penting karena MLP sensitif terhadap skala fitur
    # ----------------------------------------------------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ----------------------------------------------------------------------
    # 3. Split data menjadi train (80%) dan test (20%)
    #    - train : untuk melatih model
    #    - test  : untuk evaluasi generalisasi model
    # ----------------------------------------------------------------------
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.2, random_state=random_state
    )

    # ----------------------------------------------------------------------
    # 4. Definisi arsitektur MLP
    #    hidden_layer_sizes = (64, 32)
    #      → layer 1: 64 neuron, layer 2: 32 neuron
    #
    #    max_iter = 500 → maksimum iterasi training
    # ----------------------------------------------------------------------
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=random_state
    )

    # ----------------------------------------------------------------------
    # 5. Training model pada data training
    # ----------------------------------------------------------------------
    model.fit(Xtr, ytr)

    # ----------------------------------------------------------------------
    # 6. Evaluasi model pada data test menggunakan skor R^2
    #    R^2 = 1 → prediksi sempurna
    #    R^2 = 0 → sama seperti rata-rata
    #    R^2 < 0 → performa buruk
    # ----------------------------------------------------------------------
    score = model.score(Xte, yte)

    return model, scaler, score
