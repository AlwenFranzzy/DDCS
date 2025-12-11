# qlearning_module.py
import numpy as np

def discretize_state(pdam_used, recycled, env, n_bins_recycled=20):
    """
    Melakukan diskretisasi state continuous menjadi indeks diskrit agar bisa
    digunakan pada Q-table.
    
    Parameter:
    - pdam_used       : jumlah PDAM yang digunakan pada langkah sebelumnya (float)
    - recycled        : jumlah recycled water saat ini
    - env             : environment untuk membaca kapasitas tank
    - n_bins_recycled : jumlah bin diskret untuk variabel recycled

    Output:
    - (recycled_bin, pdam_flag)
      recycled_bin : indeks bin hasil diskretisasi stok air daur ulang
      pdam_flag    : 1 jika sebelumnya menggunakan PDAM, 0 jika tidak
    """

    # Clamp recycled agar tetap dalam kapasitas
    recycled = max(0.0, min(env.tank_capacity, recycled))

    # Hitung ukuran tiap bin
    bin_size = env.tank_capacity / n_bins_recycled

    # Tentukan posisi recycled pada bin
    recycled_bin = int(recycled / bin_size)
    if recycled_bin >= n_bins_recycled:
        recycled_bin = n_bins_recycled - 1

    # Flag apakah PDAM digunakan pada step sebelumnya
    pdam_flag = 1 if pdam_used > 0 else 0

    return (recycled_bin, pdam_flag)


def q_learning_model_assisted(env, mlp_model, scaler, episodes=10000, steps_per_episode=24,
                              alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.05):
    """
    Q-learning dengan bantuan model MLP (model-assisted RL).
    Model digunakan untuk memberikan shaping reward tambahan berdasarkan prediksi
    peningkatan stok recycled water.

    Parameter:
    - env             : environment simulasi
    - mlp_model       : model MLP yang memprediksi next_recycled
    - scaler          : scaler untuk menormalisasi fitur input MLP
    - episodes        : jumlah episode training
    - steps_per_episode : jumlah langkah setiap episode (default 24 jam)
    - alpha           : learning rate
    - gamma           : discount factor
    - epsilon_start   : nilai awal epsilon (exploration)
    - epsilon_min     : nilai minimal epsilon

    Output:
    - q_table : tabel Q-learning berukuran [bins x pdam_flag x actions]
    - rewards_history : akumulasi reward setiap episode
    """

    n_bins = 20
    n_actions = 3

    # Q-table: [bin_recycled, pdam_flag, action]
    q_table = np.zeros((n_bins, 2, n_actions))

    epsilon = epsilon_start
    rewards_history = []

    for ep in range(episodes):
        # Reset environment tiap episode
        s = env.reset()
        pdam_prev, recycled, hour, day = s

        # Diskretisasi state awal
        s_idx = discretize_state(pdam_prev, recycled, env, n_bins_recycled=n_bins)

        ep_reward = 0.0

        for step in range(steps_per_episode):

            # --- Pilih aksi menggunakan epsilon-greedy ---
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)  # eksplorasi
            else:
                action = int(np.argmax(q_table[s_idx]))  # eksploitasi

            # Jalankan environment
            next_state, reward, done, info = env.step(action)
            pdam_next, recycled_next, hour_next, day_next = next_state

            # ------------------------------------------------------------------
            #          Model-Assisted Reward Shaping
            # ------------------------------------------------------------------
            # Fitur input ke MLP: kondisi sekarang + aksi + demand
            feat = [[pdam_prev, recycled, hour, day, action, info['demand']]]

            # Normalisasi fitur
            feat_scaled = scaler.transform(feat)

            # Prediksi jumlah recycled setelah aksi
            pred_next_recycled = mlp_model.predict(feat_scaled)[0]

            # Hitung pertumbuhan recycled yang diprediksi
            recycled_growth = pred_next_recycled - recycled

            # Bonus reward jika prediksi menunjukkan peningkatan recycled
            growth_bonus = max(0.0, recycled_growth / (env.tank_capacity + 1e-9)) * 1.5

            # Kombinasikan reward RL murni dengan reward shaping
            combined_reward = reward + growth_bonus

            # ------------------------------------------------------------------
            #                       Q-table Update
            # ------------------------------------------------------------------
            ns_idx = discretize_state(pdam_next, recycled_next, env, n_bins_recycled=n_bins)

            old_q = q_table[s_idx + (action,)]
            td_target = combined_reward + gamma * np.max(q_table[ns_idx])

            # Update rule Q-learning
            q_table[s_idx + (action,)] = old_q + alpha * (td_target - old_q)

            # Update state
            pdam_prev, recycled, hour, day = pdam_next, recycled_next, hour_next, day_next
            s_idx = ns_idx
            ep_reward += combined_reward

        # Kurangi epsilon setiap episode (exploration decay)
        epsilon = max(epsilon_min, epsilon * 0.995)

        # Simpan total reward episode
        rewards_history.append(ep_reward)

    return q_table, rewards_history


def get_greedy_policy_from_qtable(q_table, env):
    """
    Menghasilkan policy deterministik (greedy) dari Q-table,
    yaitu memilih aksi bernilai Q tertinggi untuk setiap state.
    """

    def policy(state, _env=None):
        pdam_prev, recycled, hour, day = state
        idx = discretize_state(pdam_prev, recycled, env, n_bins_recycled=q_table.shape[0])
        return int(np.argmax(q_table[idx]))

    return policy


def simulate_policy(env, policy_fn, total_hours=24*7):
    """
    Simulasi policy dalam environment untuk jangka waktu tertentu.
    Menghasilkan statistik total penggunaan PDAM, recycled, dan trace level tank.

    Output dictionary berisi:
    - pdam_total
    - recycled_used_total
    - recycled_trace
    - pdam_used_list
    - recycled_used_list
    - time_steps
    """

    state = env.reset()
    pdam_prev, recycled, hour, day = state

    pdam_total = 0.0
    recycled_total_used = 0.0
    recycled_trace = []

    # List untuk visualisasi
    pdam_used_list = []
    recycled_used_list = []
    time_steps = []

    for t in range(total_hours):
        # Pilih aksi berdasarkan policy
        action = policy_fn(state, env)

        # Jalankan environment
        next_state, reward, done, info = env.step(action)

        # Ambil data penggunaan air
        pdam_used = info['pdam_used']
        recycled_used = info['recycled_used']

        # Akumulasikan statistik
        pdam_total += pdam_used
        recycled_total_used += recycled_used
        recycled_trace.append(next_state[1])  # stok recycled tiap waktu

        # Simpan untuk plotting
        pdam_used_list.append(pdam_used)
        recycled_used_list.append(recycled_used)
        time_steps.append(t)

        state = next_state

    return {
        "pdam_total": pdam_total,
        "recycled_used_total": recycled_total_used,
        "recycled_trace": recycled_trace,
        "pdam_used_list": pdam_used_list,
        "recycled_used_list": recycled_used_list,
        "time_steps": time_steps,
    }
