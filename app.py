import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Fermion Lattice Refrigeration", layout="wide")

st.title(r"Fermionic Lattice: Entropy & Refrigeration")

# Sidebar Inputs
st.sidebar.header("Global Parameters")
T_user = st.sidebar.slider("Temperature (T)", 0.01, 0.5, 0.05, 0.01)
U_user = st.sidebar.slider("Interaction Energy (U)", -1.0, 1.0, 0.3, 0.05)
mu_fixed = st.sidebar.slider("Fixed Chemical Potential (μ) for S vs T", -0.5, 1.0, 0.5, 0.05)

def calculate_physics(T_input, U_input, mu_input, g=1.0):
    # Ensure T is at least a very small positive number to avoid div by zero
    T = np.atleast_1d(np.maximum(T_input, 1e-6))
    mu = np.atleast_1d(mu_input)
    beta = 1.0 / T
    
    states = [(0, 0, 1), (0, 1, 2), (U_input, 2, 1), (g + U_input, 3, 2)]
    
    weights = []
    omegas = []
    for E, N, deg in states:
        omega_i = E - mu * N
        # Use log-sum-exp trick implicitly or clipping to prevent overflow
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

# --- 1. Calculations for the mu-sweep (fixed T) ---
mu_range = np.linspace(-0.5, 1.0, 300)
p_mu, s_mu = calculate_physics(T_user, U_user, mu_range)

# --- 2. Calculations for the T-sweep (fixed mu) ---
T_range = np.linspace(0.001, 0.5, 500) # Updated range 0 to 0.5
U_lines = [0.4, 0.2, 0.1]
s_vs_t_results = {}
for u_val in U_lines:
    _, s_t = calculate_physics(T_range, u_val, mu_fixed)
    s_vs_t_results[u_val] = s_t

# --- LAYOUT & PLOTTING ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(r"Probabilities & Entropy vs $\mu$")
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    for i in range(4):
        ax1.plot(mu_range, p_mu[i], label=f'P{i}', lw=2)
    ax1.axvline(mu_fixed, color='black', linestyle='--', alpha=0.4)
    ax1.set_ylabel("Probability")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)
    
    ax2.plot(mu_range, s_mu, color='purple', lw=2)
    ax2.axvline(mu_fixed, color='black', linestyle='--', alpha=0.4)
    ax2.set_ylabel("Entropy $S$")
    ax2.set_xlabel(r"Chemical Potential $\mu$")
    ax2.grid(True, alpha=0.2)
    st.pyplot(fig1)

with col2:
    st.subheader(f"Log-Scale Entropy vs $T$ at $\mu = {mu_fixed}$")
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    
    for u_v in U_lines:
        ax3.plot(T_range, s_vs_t_results[u_v], label=f'U = {u_v}', lw=2)
    
    # User-selected point highlight
    _, current_s = calculate_physics(T_user, U_user, mu_fixed)
    ax3.scatter([T_user], [current_s], color='red', zorder=5, label=f'Current (U={U_user})')
    
    ax3.set_yscale('log')
    ax3.set_ylim(0.01, 2.0) # Requested range
    ax3.set_xlim(0, 0.5)
    ax3.set_xlabel("Temperature $T$")
    ax3.set_ylabel("Entropy $S$ (Log Scale)")
    ax3.set_title("Cooling Capacity Analysis")
    ax3.legend()
    ax3.grid(True, which="both", alpha=0.2) # Show minor gridlines for log scale
    st.pyplot(fig2)

    # Metrics
    idx = (np.abs(mu_range - mu_fixed)).argmin()
    st.info(f"**Live Values at μ = {mu_fixed}, T = {T_user}:**")
    m_cols = st.columns(5)
    for i in range(4):
        m_cols[i].metric(f"P{i}", f"{p_mu[i][idx]:.3f}")
    m_cols[4].metric("Entropy S", f"{s_mu[idx]:.3f}")
