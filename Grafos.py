import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(layout="wide")

# ====== CONFIGURACIÓN ESTÉTICA ======
COLOR_FONDO = "#F8F9FA"
COLOR_NODO_NORMAL = "#AED6F1"
COLOR_INICIO = "#2ECC71"
COLOR_OBJETIVO = "#F1C40F"
COLOR_RUTA = "#E74C3C"
COLOR_BORDES = "#2C3E50"

LETRAS = ["A","B","C","D","E","F","G","H","I","J",
          "K","L","M","N","Ñ","O","P","Q","R","S","T","U"]

st.title("Visualización de Ruta Óptima (Q-Learning)")

# ====== MATRIZ BASE ======
R_base = np.array([
[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
])

# ====== SELECCIÓN INTERACTIVA ======
col1, col2 = st.columns(2)

with col1:
    letra_inicio = st.selectbox("Selecciona nodo de inicio:", LETRAS)

with col2:
    letra_objetivo = st.selectbox("Selecciona nodo objetivo:", LETRAS, index=10)

inicio = LETRAS.index(letra_inicio)
objetivo = LETRAS.index(letra_objetivo)

# ====== ACTUALIZAR RECOMPENSAS ======
R = R_base.copy()
for i in range(22):
    if R[i, objetivo] > 0:
        R[i, objetivo] = 100

# ====== Q-LEARNING ======
gamma = 0.75
alpha = 0.9
Q = np.zeros((22,22))

for _ in range(1000):
    estado = np.random.randint(0,22)
    acciones = np.where(R[estado] > 0)[0]
    if len(acciones) == 0:
        continue
    accion = np.random.choice(acciones)
    siguiente = accion
    TD = R[estado, accion] + gamma * np.max(Q[siguiente]) - Q[estado, accion]
    Q[estado, accion] += alpha * TD

# ====== OBTENER RUTA ======
def obtener_ruta(Q, R, inicio, objetivo, max_pasos=60):
    estado = inicio
    ruta = [estado]
    visitados = {estado}
    for _ in range(max_pasos):
        if estado == objetivo:
            break
        acciones = np.where(R[estado] > 0)[0]
        if len(acciones) == 0:
            break
        siguiente = acciones[np.argmax(Q[estado, acciones])]
        if siguiente in visitados:
            break
        ruta.append(siguiente)
        visitados.add(siguiente)
        estado = siguiente
    return ruta

ruta = obtener_ruta(Q, R, inicio, objetivo)
ruta_edges = list(zip(ruta[:-1], ruta[1:]))
ruta_set = set(ruta_edges)

# ====== GRAFO ======
G = nx.DiGraph()
for i in range(22):
    G.add_node(i)

for i in range(22):
    for j in range(22):
        if R_base[i,j] > 0:
            G.add_edge(i,j)

pos = nx.kamada_kawai_layout(G)

# ====== VISUALIZACIÓN ======
fig, ax = plt.subplots(figsize=(14,8), facecolor=COLOR_FONDO)
ax.set_facecolor(COLOR_FONDO)

edges_no_ruta = [e for e in G.edges() if e not in ruta_set]

nx.draw_networkx_edges(
    G, pos, edgelist=edges_no_ruta,
    edge_color="#BDC3C7", alpha=0.3, width=1,
    arrowsize=10, connectionstyle="arc3,rad=0.1", ax=ax
)

nx.draw_networkx_edges(
    G, pos, edgelist=ruta_edges,
    edge_color=COLOR_RUTA, width=4, alpha=0.8,
    arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax
)

node_colors = [
    COLOR_INICIO if n==inicio
    else COLOR_OBJETIVO if n==objetivo
    else COLOR_NODO_NORMAL
    for n in G.nodes()
]

nx.draw_networkx_nodes(
    G, pos,
    node_size=1000,
    node_color=node_colors,
    edgecolors=COLOR_BORDES,
    linewidths=1.5,
    ax=ax
)

nx.draw_networkx_labels(
    G, pos,
    {i: LETRAS[i] for i in range(len(LETRAS))},
    font_size=10,
    font_weight="bold"
)

# Numeración pasos
for i, (u, v) in enumerate(ruta_edges):
    mid_x = (pos[u][0] + pos[v][0]) / 2
    mid_y = (pos[u][1] + pos[v][1]) / 2
    ax.text(mid_x, mid_y + 0.05, str(i+1),
            color="white", weight="bold",
            bbox=dict(facecolor=COLOR_RUTA,
                      edgecolor="none",
                      boxstyle="circle,pad=0.2"))

# Reward label
if ruta_edges:
    u100 = ruta_edges[-1][0]
    target_mid_x = (pos[u100][0] + pos[objetivo][0]) / 2
    target_mid_y = (pos[u100][1] + pos[objetivo][1]) / 2

    ax.text(target_mid_x, target_mid_y - 0.08,
            "REWARD +100",
            color="black", weight="bold",
            bbox=dict(facecolor=COLOR_OBJETIVO,
                      alpha=0.8,
                      boxstyle="round,pad=0.3"))

plt.title("Visualización de Ruta Óptima (Q-Learning)",
          fontsize=16, fontweight="bold", pad=20)

ax.axis("off")
plt.tight_layout()

st.pyplot(fig)


