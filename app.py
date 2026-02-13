import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Fermion Lattice Refrigeration", layout="wide")

st.title(r"Fermionic Lattice: Entropy & Refrigeration")
st.markdown("Analyze the thermodynamics of a 2-component fermionic system.")

# Sidebar for interactive controls
st.sidebar.header("Global Parameters")
T_user = st.sidebar.slider("Temperature (T)", 0.01, 1.0, 0.05, 0.01)
U_user = st.sidebar.slider("Interaction Energy (U)", -1.0, 1.0, 0.3, 0.05)
mu_fixed = st.sidebar.slider("Fixed Chemical Potential (μ) for S vs T", -0.5, 1.0, 0.5, 0.05)

def calculate_physics(T_input, U_input, mu_input, g=1.0):
    """Calculates state probabilities and entropy for given T, U, and mu."""
    T = np.atleast_1d(np.maximum(T_input, 1e-6))
    mu = np.atleast_1d(mu_input)
    beta = 1.0 / T
    
    # State data: (Energy, Particles, Multiplicity)
    states = [(0, 0, 1), (0, 1, 2), (U_input, 2, 1), (g + U_input, 3, 2)]
    
    weights = []
    omegas = []
    for E, N, deg in states:
        omega_i = E - mu * N
        # Numerical stability: clip to prevent overflow/underflow
        w = deg * np.exp(np.clip(-beta * omega_i, -500, 500))
        weights.append(w)
        omegas.append(omega_i)
    
    weights = np.array(weights)
    omegas = np.array(omegas)
    Z = np.sum(weights, axis=0)
    
    probs = weights / Z
    avg_omega = np.sum(weights * omegas, axis=0) / Z
    entropy = beta * avg_omega + np.log(Z)
    
    return probs, entropy

# --- 1. Calculations for Mu-sweep ---
mu_range = np.linspace(-0.5, 1.0, 300)
p_mu, s_mu = calculate_physics(T_user, U_user, mu_range)

# --- 2. Calculations for Log-Log T-sweep ---
T_log_range = np.logspace(np.log10(0.01), np.log10(1.0), 500)
U_lines = [0.4, 0.0, -0.4]
s_vs_t_results = {}
for u_val in U_lines:
    _, s_t = calculate_physics(T_log_range, u_val, mu_fixed)
    s_vs_t_results[u_val] = s_t

# --- Plotting ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(r"Probabilities & Entropy vs $\mu$")
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    # Probabilities
    labels = [r'$P_0$', r'$P_1$', r'$P_2$', r'$P_3$']
    for i in range(4):
        ax1.plot(mu_range, p_mu[i], label=labels[i], lw=2)
    ax1.axvline(mu_fixed, color='black', linestyle='--', alpha=0.4)
    ax1.set_ylabel("Probability")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)
    
    # Entropy
    ax2.plot(mu_range, s_mu, color='purple', lw=2)
    ax2.axvline(mu_fixed, color='black', linestyle='--', alpha=0.4)
    ax2.set_ylabel(r"Entropy $S$")
    ax2.set_xlabel(r"Chemical Potential $\mu$")
    ax2.grid(True, alpha=0.2)
    st.pyplot(fig1)

with col2:
    st.subheader(f"Log-Log Entropy vs T at mu = {mu_fixed}")
    fig2, ax3 = plt.subplots(figsize=(8, 6.5))
    
    colors = ['#d62728', '#1f77b4', '#2ca02c'] # U=0.4 (Red), U=0 (Blue), U=-0.4 (Green)
    for u_v, color in zip(U_lines, colors):
        ax3.plot(T_log_range, s_vs_t_results[u_v], label=f'U = {u_v}', lw=2.5, color=color)
    
    # Scalar extraction for current point
    _, current_s_arr = calculate_physics(T_user, U_user, mu_fixed)
    current_s_val = current_s_arr.item()
    
    ax3.scatter([T_user], [current_s_val], color='black', s=80, edgecolors='white', zorder=5, label='Current Point')
    
    # Formatting
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim(0.01, 1.0)
    ax3.set_ylim(0.001, 2.0)
    
    ax3.set_xlabel("Temperature T (Log Scale)")
    ax3.set_ylabel("Entropy S (Log Scale)")
    ax3.legend()
    ax3.grid(True, which="both", alpha=0.3)
    st.pyplot(fig2)

    st.info(f"Thermodynamic metrics for μ = {mu_fixed}")
    st.write(f"Entropy $S$ at current $U={U_user}$: **{current_s_val:.4f}**")
