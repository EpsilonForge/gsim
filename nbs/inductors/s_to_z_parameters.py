import matplotlib.pyplot as plt
import numpy as np
import skrf as rf


def plot_z_parameters(results, freq_unit="GHz", title_suffix=""):
    """
    Plot Z-parameters in log-log scale from simulation results.
    Works for any NxN S-parameter network (2, 4, etc. ports).

    Parameters
    ----------
    results : simulation results object with .freq, .port_names, and [(pi,pj)].complex
    freq_unit : str, frequency unit for x-axis label ("GHz" or "MHz")
    title_suffix : str, optional suffix for plot titles
    """
    # ── 1. Build frequency array ──────────────────────────────────────────────
    freq_scale = {"GHz": 1e9, "MHz": 1e6}
    f = results.freq * freq_scale.get(freq_unit, 1e9)
    w = 2 * np.pi * f

    ports = results.port_names
    n = len(ports)
    print(f"Detected {n} ports: {ports}")

    # ── 2. Build S-matrix ─────────────────────────────────────────────────────
    S = np.zeros((len(f), n, n), dtype=complex)
    for i, pi in enumerate(ports):
        for j, pj in enumerate(ports):
            S[:, i, j] = results[(pi, pj)].complex

    ntwk = rf.Network(f=f, s=S, f_unit="hz")
    Z = ntwk.z
    Y = ntwk.y
    f_plot = f / freq_scale.get(freq_unit, 1e9)

    # ── 3. Plot full Z-parameter matrix ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.tab10
    colors = [cmap(k) for k in np.linspace(0, 1, n * n)]

    idx = 0
    for i in range(n):
        for j in range(n):
            label = f"|Z{i + 1}{j + 1}|"
            ls = "-" if i == j else ("--" if i < j else ":")
            ax.plot(
                f_plot,
                np.abs(Z[:, i, j]),
                label=label,
                linestyle=ls,
                linewidth=2 if i == j else 1.2,
                color=colors[idx],
            )
            idx += 1

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Frequency ({freq_unit})")
    ax.set_ylabel("Magnitude |Z| (Ω)")
    ax.set_title(f"Z-Parameters Log-Log {title_suffix}")
    ax.legend(ncol=n, fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return Z, Y, f


def plot_differential_z_parameters(results, freq_unit="GHz", title_suffix=""):
    """
    Calculate and plot mixed-mode Z-parameters (Differential, Common, etc.)
    in log-log scale.

    Assumes the network is a 4-port configured as two differential pairs.
    """
    # ── 1. Data preparation ───────────────────────────────────────────────────
    freq_scale = {"GHz": 1e9, "MHz": 1e6}
    f = results.freq * freq_scale.get(freq_unit, 1e9)
    ports = results.port_names
    n = len(ports)

    S = np.zeros((len(f), n, n), dtype=complex)
    for i, pi in enumerate(ports):
        for j, pj in enumerate(ports):
            S[:, i, j] = results[(pi, pj)].complex

    # Create network and convert to mixed-mode
    ntwk = rf.Network(f=f, s=S, f_unit="hz")
    ntwk.se2gmm(p=int(n / 2))
    Z_mm = ntwk.z
    f_plot = f / freq_scale.get(freq_unit, 1e9)

    # ── 2. Plotting ───────────────────────────────────────────────────────────
    # Mixed-mode port names are usually:
    # 1d (diff), 2d (diff), 1c (common), 2c (common)
    labels_mm = [
        "$Z_{dd11}$",
        "$Z_{dd12}$",
        "$Z_{dd21}$",
        "$Z_{dd22}$",
        "$Z_{cc11}$",
        "$Z_{cc12}$",
        "$Z_{cc21}$",
        "$Z_{cc22}$",
    ]
    # Note: The matrix is 4x4, elements of interest can be plotted.

    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot the main differential terms (DD) if they exist
    # (Usually Zdd is the top-left 2x2 submatrix)
    for i in range(2):
        for j in range(2):
            ax.plot(
                f_plot,
                np.abs(Z_mm[:, i, j]),
                label=f"$|Z_{{dd{i + 1}{j + 1}}}|$",
                linestyle="-",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Frequency ({freq_unit})")
    ax.set_ylabel("Magnitude |Z_diff| (Ω)")
    ax.set_title(f"Differential Z-Parameters (Mixed-Mode) {title_suffix}")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    return Z_mm, f
