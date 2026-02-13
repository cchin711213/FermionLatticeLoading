import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Fermion Lattice Refrigeration", layout="wide")

st.title(r"Fermionic Lattice: Entropy & Refrigeration")

# Sidebar Inputs
st.sidebar.header("Global Parameters")
T_user = st.sidebar.slider("Temperature (T)", 0.01, 1.0, 0.05, 0.01)
U_user = st.sidebar.slider("Interaction Energy (U)", -1.0, 1.0, 0.3, 0.05)
mu_fixed = st.sidebar.slider("Fixed Chemical Potential (μ) for S vs T", -0.5, 1.0, 0.5, 0.05)

def get_physics(T, U, mu_vals, g=1.0):
    beta = 1.0 / T
    states = [(0, 0, 1), (0, 1, 2), (U, 2, 1), (g + U, 3, 2)]
    
    p_res = {f'P{i}': [] for i in range(4)}
    s_res = []
    
    for mu in mu_vals:
        omegas = [E - mu * N for E, N, deg in states]
        weights = [deg * np.exp(-beta * o) for deg, o in zip([s[2] for s in states], omegas)]
        Z = sum(weights)
        for i in range(4):
            p_res[f'P{i}'].append(weights[i] / Z)
        
        avg_omega = sum(w * o for w, o in zip(weights, omegas)) / Z
        s_res.append(beta * avg_omega + np.log(Z))
    return p_res, s_res

# Main Calculations
mu_range = np.linspace(-0.5, 1.0, 300)
probs, entropy_mu = get_physics(T_user, U_user, mu_range)

# S vs T Calculation at fixed mu
T_range = np.linspace(0.01, 0.8, 200)
U_lines = [0.4, 0.2, 0.1]
s_vs_t_data = {}
for u_val in U_lines:
    _, s_t = get_physics(T_range, u_val, [mu_fixed])
    s_vs_t_data[u_val] = s_t

# --- LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("State Probabilities & Entropy vs $\mu$")
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    for i, (lab, p) in enumerate(probs.items()):
        ax1.plot(mu_range, p, label=lab, lw=2)
    ax1.axvline(mu_fixed, color='k', linestyle='--', alpha=0.5, label=f'Selected μ={mu_fixed}')
    ax1.set_ylabel("Probability")
    ax1.legend()
    
    ax2.plot(mu_range, entropy_mu, color='purple', lw=2)
    ax2.axvline(mu_fixed, color='k', linestyle='--', alpha=0.5)
    ax2.set_ylabel("Entropy $S$")
    ax2.set_xlabel(r"Chemical Potential $\mu$")
    st.pyplot(fig1)

with col2:
    st.subheader(f"Entropy vs Temperature at $\mu = {mu_fixed}$")
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    
    for u_v in U_lines:
        ax3.plot(T_range, s_vs_t_data[u_v], label=f'U = {u_v}', lw=2)
    
    ax3.set_xlabel("Temperature $T$")
    ax3.set_ylabel("Entropy $S$")
    ax3.set_title(r"Comparison of $S(T)$ for cooling cycles")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Display precise values for the sidebar-selected point
    idx = (np.abs(mu_range - mu_fixed)).argmin()
    st.info(f"**At μ = {mu_fixed}, T = {T_user}, U = {U_user}:**")
    cols = st.columns(5)
    for i in range(4):
        cols[i].metric(f"P{i}", f"{probs[f'P{i}'][idx]:.3f}")
    cols[4].metric("Entropy S", f"{entropy_mu[idx]:.3f}")
