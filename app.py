import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Fermionic Lattice Model", layout="wide")

st.title(r"2-Component Fermionic Lattice Model")
st.markdown(r"""
This app calculates the probabilities $P_i$ and the Entropy $S$ of a 2-component fermionic system 
in a grand canonical ensemble.
""")

# Sidebar for user inputs
st.sidebar.header("Parameters")
T = st.sidebar.slider("Temperature (T)", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
U = st.sidebar.slider("Interaction Energy (U)", min_value=-1.0, max_value=1.0, value=0.3, step=0.05)
g = 1.0  # Fixed gap as per instructions

# Calculation function
def calculate_lattice_physics(T, U, mu_min=-0.5, mu_max=1.0, g=1.0):
    beta = 1.0 / T
    mu_range = np.linspace(mu_min, mu_max, 500)
    
    # States: (Energy E, Particle Number N, Multiplicity g_i)
    states = [
        (0, 0, 1),      # P0
        (0, 1, 2),      # P1
        (U, 2, 1),      # P2
        (g + U, 3, 2)   # P3
    ]
    
    results = {f'P{i}': [] for i in range(4)}
    entropy_list = []
    
    for mu in mu_range:
        weights = []
        omegas = [] 
        for E, N, deg in states:
            omega_i = E - mu * N
            w = deg * np.exp(-beta * omega_i)
            weights.append(w)
            omegas.append(omega_i)
            
        Z = sum(weights)
        
        for i in range(4):
            results[f'P{i}'].append(weights[i] / Z)
            
        avg_omega = sum(weights[i] * omegas[i] for i in range(4)) / Z
        S = beta * avg_omega + np.log(Z)
        entropy_list.append(S)
        
    return mu_range, results, entropy_list

# Execute calculation
mu_axis, probs, entropy_axis = calculate_lattice_physics(T, U)

# Plotting with Matplotlib
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot 1: Probabilities
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, (label, p_values) in enumerate(probs.items()):
    ax1.plot(mu_axis, p_values, label=label, lw=2.5, color=colors[i])

ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title(fr'State Probabilities ($T={T}$, $U={U}$)', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Entropy
ax2.plot(mu_axis, entropy_axis, color='purple', lw=2, label='Entropy $S$')
ax2.set_xlabel(r'Chemical Potential ($\mu$)', fontsize=12)
ax2.set_ylabel(r'Entropy $S$', fontsize=12)
ax2.set_title(r'Entropy vs Chemical Potential', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

# Display in Streamlit
st.pyplot(fig)

# Data Table (Optional)
if st.checkbox("Show raw data"):
    st.write("First 10 values of the calculation:")
    st.write(np.column_stack([mu_axis[:10], entropy_axis[:10]]))
