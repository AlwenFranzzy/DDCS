# data_module.py
# ---------------------------------------------------------------
# Modul untuk melakukan data collection dari environment.
# Sistem menjalankan environment dengan aksi acak
# agar menghasilkan dataset berisi transisi state-action-next_state.
# Dataset ini nantinya digunakan untuk training model MLP.
# ---------------------------------------------------------------

import random
import pandas as pd

def collect_data(env, n_steps=2000, seed=0):
    """
    Mengumpulkan data dari environment menggunakan aksi acak.
    Tiap langkah menghasilkan satu baris data berisi:
    - pdam_prev        : jumlah air PDAM yang dipakai pada langkah sebelumnya
    - recycled_before  : level air daur ulang sebelum aksi
    - hour             : jam simulasi (0–23)
    - day              : hari simulasi (0–6)
    - action           : aksi yang diambil (0 = HOLD, 1 = RELEASE, 2 = WATER_PLANTS)
    - demand           : kebutuhan air pada langkah tersebut
    - next_recycled    : level air daur ulang setelah aksi dilakukan
    - pdam_used        : jumlah air PDAM yang dipakai oleh sistem
    - recycled_used    : jumlah air daur ulang yang dipakai oleh sistem

    Parameter:
    -----------
    env : WaterReuseEnv
        Environment simulasi sistem air limbah.
    n_steps : int
        Jumlah langkah simulasi untuk data collection.
    seed : int
        Seed untuk memastikan random action reproducible.

    Return:
    -------
    df : pandas.DataFrame
        Data transisi berukuran n_steps x 9 kolom.
    """

    # Set seed agar aksi acak tetap konsisten (reproducible)
    random.seed(seed)

    # Reset environment untuk mendapatkan state awal
    state = env.reset()

    # List penampung baris data
    records = []

    # Loop utama untuk mengumpulkan n_steps data
    for _ in range(n_steps):

        # -------------------------------------------------------
        # 1. Pilih aksi random: 0 = HOLD, 1 = RELEASE, 2 = WATER
        # -------------------------------------------------------
        action = random.choice([0, 1, 2])

        # -------------------------------------------------------
        # 2. Jalankan satu langkah simulasi
        #    next_state : state berikutnya
        #    reward     : reward dari lingkungan
        #    done       : status episode berakhir
        #    info       : dictionary berisi demand, pdam_used, recycled_used, dll
        # -------------------------------------------------------
        next_state, reward, done, info = env.step(action)

        # -------------------------------------------------------
        # 3. Pecah state saat ini
        #    Format state dari env adalah:
        #    (pdam_prev, recycled_before, hour, day)
        # -------------------------------------------------------
        pdam_prev, recycled_before, hour, day = state

        # Level recycled setelah aksi (untuk label prediksi model)
        next_recycled = next_state[1]

        # -------------------------------------------------------
        # 4. Simpan satu record data
        # -------------------------------------------------------
        records.append([
            pdam_prev,         # penggunaan PDAM sebelumnya
            recycled_before,   # level air recycled sebelum aksi
            hour,              # jam simulasi
            day,               # hari simulasi
            action,            # aksi yang diambil
            info['demand'],    # kebutuhan air saat ini
            next_recycled,     # level recycled setelah aksi
            info['pdam_used'], # air PDAM yang dipakai sistem
            info['recycled_used'] # air recycled yang dipakai sistem
        ])

        # Update state untuk iterasi berikutnya
        state = next_state

    # -------------------------------------------------------
    # 5. Konversi list menjadi DataFrame
    # -------------------------------------------------------
    df = pd.DataFrame(records, columns=[
        'pdam_prev', 'recycled_before', 'hour', 'day',
        'action', 'demand', 'next_recycled',
        'pdam_used', 'recycled_used'
    ])

    return df
