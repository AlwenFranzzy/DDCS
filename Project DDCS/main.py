# main.py
"""
Main pipeline untuk sistem pengendali penggunaan air berbasis:
1. Pengumpulan data (random policy)
2. Pelatihan model prediksi (MLP)
3. Q-learning model-assisted
4. Simulasi kebijakan terbaik
5. Visualisasi hasil

Seluruh blok fungsi visualisasi berada pada file ini untuk memudahkan observasi.
"""

from environment import WaterReuseEnv
from data import collect_data
from model import train_mlp
from qlearning_module import q_learning_model_assisted, get_greedy_policy_from_qtable, simulate_policy
import matplotlib.pyplot as plt


# ============================================================
# 1. VISUALISASI KURVA REWARD LEARNING
# ============================================================
def plot_rewards(rewards):
    """
    Menampilkan grafik reward total per episode.
    Berfungsi untuk mengevaluasi apakah Q-learning mengalami peningkatan performa.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, linewidth=1.2)
    plt.xlabel("Episode")
    plt.ylabel("Episode total reward (shaped)")
    plt.title("Kurva Pembelajaran Q-learning")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. VISUALISASI TRAJEKTORI STOK AIR DAUR ULANG
# ============================================================
def plot_recycled_trace(trace):
    """
    Menampilkan perubahan stok air daur ulang sepanjang simulasi.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(trace, linewidth=1.2)
    plt.xlabel("Step")
    plt.ylabel("Recycled stock")
    plt.title("Perubahan Stok Air Daur Ulang")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. VISUALISASI TANGKI (AWAL – TENGAH – AKHIR)
# ============================================================
def plot_tank_tanks(time_steps, pdam_used, recycled_used,
                    title="Tank view: PDAM vs Recycled over time"):
    """
    Menampilkan 3 visualisasi tangki: awal, pertengahan, dan akhir simulasi.

    Visualisasi menampilkan proporsi PDAM vs Recycle dalam bentuk "tank chart".
    Sangat membantu untuk melihat perubahan pola penggunaan.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    time_steps = np.asarray(time_steps)
    pdam_used = np.asarray(pdam_used)
    recycled_used = np.asarray(recycled_used)

    total_steps = len(time_steps)
    if total_steps == 0:
        print("No data to plot tank view.")
        return

    # Penanganan data pendek secara aman
    if total_steps < 3:
        ranges = [(0, total_steps), (0, total_steps), (0, total_steps)]
    else:
        one_third = total_steps // 3
        two_third = 2 * total_steps // 3
        ranges = [
            (0, one_third),
            (one_third, two_third),
            (two_third, total_steps)
        ]

    # Fungsi bantu: hitung rasio PDAM & recycle
    def compute_ratio(start, end):
        if end <= start:
            return 0.5, 0.5
        p_sum = float(pdam_used[start:end].sum())
        r_sum = float(recycled_used[start:end].sum())
        total = p_sum + r_sum
        if total <= 1e-9:
            return 0.5, 0.5
        return p_sum / total, r_sum / total

    # Hitung rasio tiap fase
    ratios = []
    labels = ["Awal simulasi", "Pertengahan", "Akhir simulasi"]
    for (s, e), lab in zip(ranges, labels):
        p_r, r_r = compute_ratio(s, e)
        ratios.append((p_r, r_r, lab))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    tank_width = 0.5
    gap = 0.7
    base_x = [0, tank_width + gap, 2 * (tank_width + gap)]
    tank_height = 1.0

    for i, (p_r, r_r, label) in enumerate(ratios):
        x_left = base_x[i]
        x_right = x_left + tank_width

        # Outline
        ax.plot([x_left, x_right], [0, 0], color="black")
        ax.plot([x_left, x_left], [0, tank_height], color="black")
        ax.plot([x_right, x_right], [0, tank_height], color="black")
        ax.plot([x_left, x_right], [tank_height, tank_height], color="black")

        # Tinggi proporsi
        p_h = p_r * tank_height
        r_h = r_r * tank_height

        # Isi PDAM (biru)
        ax.fill_between([x_left, x_right], 0, p_h, alpha=0.8, color="tab:blue",
                        label="PDAM" if i == 0 else None)

        # Isi Recycle (hijau)
        ax.fill_between([x_left, x_right], p_h, p_h + r_h, alpha=0.8, color="tab:green",
                        label="Recycle" if i == 0 else None)

        # Label bawah tangki
        ax.text((x_left + x_right) / 2, -0.15, label, ha="center", va="top")

        # Label persentase
        ax.text((x_left + x_right) / 2, max(p_h / 2, 0.02),
                f"{p_r*100:.1f}% PDAM", ha="center", color="white")
        ax.text((x_left + x_right) / 2, p_h + max(r_h / 2, 0.02),
                f"{r_r*100:.1f}% Recycle", ha="center")

    ax.set_xlim(-0.5, base_x[-1] + tank_width + 0.5)
    ax.set_ylim(-0.3, 1.2)
    ax.set_xticks([])
    ax.set_ylabel("Proporsi isi tangki")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. MAIN PIPELINE
# ============================================================
def main():
    """
    Pipeline utama dari project.
    """

    # --------------------------------------------------------
    # 1. Inisialisasi Environment
    # --------------------------------------------------------
    env = WaterReuseEnv(
        tank_capacity=200.0,
        initial_recycled=50.0,
        treatment_rate=0.6,
        seed=42
    )

    # --------------------------------------------------------
    # 2. Collect data (random policy)
    # --------------------------------------------------------
    print(">> Collecting data (random actions)...")
    df = collect_data(env, n_steps=3000, seed=42)
    print("Collected rows:", len(df))

    # --------------------------------------------------------
    # 3. Train MLP prediction model
    # --------------------------------------------------------
    print(">> Training MLP model to predict next_recycled...")
    model, scaler, score = train_mlp(df)
    print("MLP R^2 score (test):", score)

    # --------------------------------------------------------
    # 4. Train Q-learning with model-assisted reward shaping
    # --------------------------------------------------------
    print(">> Training Q-learning with model-assisted shaping...")
    q_table, rewards = q_learning_model_assisted(
        env, model, scaler,
        episodes=10000,
        steps_per_episode=24,  # 1 hari
        alpha=0.1,
        gamma=0.99
    )
    print("Training done. Last 10 rewards:", rewards[-10:])

    # Grafik reward
    plot_rewards(rewards)

    # --------------------------------------------------------
    # 5. Ambil kebijakan terbaik (greedy policy)
    # --------------------------------------------------------
    policy = get_greedy_policy_from_qtable(q_table, env)

    # --------------------------------------------------------
    # 6. Simulasi kebijakan selama 7 hari
    # --------------------------------------------------------
    print(">> Simulating learned policy for 7 days...")
    sim = simulate_policy(env, policy, total_hours=24 * 7)

    print("PDAM total used in 7 days:", sim['pdam_total'])
    print("Recycled used total:", sim['recycled_used_total'])

    # --------------------------------------------------------
    # 7. Visualisasi hasil simulasi
    # --------------------------------------------------------
    plot_recycled_trace(sim['recycled_trace'])

    plot_tank_tanks(
        sim['time_steps'],
        sim['pdam_used_list'],
        sim['recycled_used_list'],
        title="Visualisasi Tangki: Perbandingan PDAM vs Air Daur Ulang (Awal–Akhir Simulasi)"
    )


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
