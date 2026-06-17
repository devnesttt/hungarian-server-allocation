"""
Alokasi Server Cloud Kampus — Algoritma Hungarian
Riset Operasi SIS3532 | Kelompok 10 | UNSRAT
Deploy: Streamlit Cloud
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, math
from itertools import permutations
from copy import deepcopy

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alokasi Server Cloud — Hungarian",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark theme override */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b26; border-right: 1px solid #2a3550; }
[data-testid="stHeader"]           { background: #161b26; border-bottom: 1px solid #2a3550; }

h1,h2,h3,h4 { color: #dde4f0 !important; }
p, li, label { color: #8b9bbf !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b26;
    border: 1px solid #2a3550;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"]  { color: #4f8ef7 !important; font-family: monospace; }
[data-testid="stMetricLabel"]  { color: #8b9bbf !important; font-size: 11px; }

/* Step expander */
[data-testid="stExpander"] {
    background: #161b26;
    border: 1px solid #2a3550;
    border-radius: 8px;
    margin-bottom: 8px;
}

/* Buttons */
.stButton > button {
    background: #2563eb;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover { background: #1d4ed8; }

/* Info / success / warning boxes */
.info-box {
    background: rgba(79,142,247,.1);
    border: 1px solid rgba(79,142,247,.3);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #8b9bbf;
    font-size: 13px;
    line-height: 1.8;
}
.result-box {
    background: rgba(34,197,94,.08);
    border: 1px solid rgba(34,197,94,.3);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
}
.critical-box {
    background: rgba(239,68,68,.08);
    border: 1px solid rgba(239,68,68,.3);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
}
.step-desc {
    background: #1c2333;
    border-left: 3px solid #4f8ef7;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 12px;
    color: #8b9bbf;
    margin-bottom: 10px;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════
ALL_SERVICES = [
    "Portal Akademik","E-Learning UNSRAT","SIMPEG","Perpustakaan Digital",
    "Email Kampus","SSO Auth","Sistem Keuangan","Absensi Online",
    "Sistem Registrasi","Jurnal Online","Video Conference","Portal Alumni",
    "Sistem Beasiswa","Helpdesk IT","Cloud Storage","Analytics & BI",
    "API Gateway","Net Monitoring","Backup Service","CDN Service",
]
ALL_SERVERS = [
    "Klaster-A","Klaster-B","Klaster-C","Klaster-D","Klaster-E",
    "Klaster-F","Klaster-G","Klaster-H","Klaster-I","Klaster-J",
    "Klaster-K","Klaster-L","Klaster-M","Klaster-N","Klaster-O",
    "Klaster-P","Klaster-Q","Klaster-R","Klaster-S","Klaster-T",
]

# ══════════════════════════════════════════════════════════════════════
# HUNGARIAN ALGORITHM
# ══════════════════════════════════════════════════════════════════════
class HungarianAlgorithm:
    def __init__(self, matrix: np.ndarray, mode: str = "min"):
        self.original   = matrix.copy().astype(float)
        self.n          = matrix.shape[0]
        self.mode       = mode
        self.steps      = []
        self.assignment = None
        self.total      = None

    def _augment(self, r, mat, row_a, col_a, tried):
        for c in range(self.n):
            if mat[r, c] == 0 and c not in tried:
                tried.add(c)
                if col_a[c] == -1 or self._augment(col_a[c], mat, row_a, col_a, tried):
                    row_a[r] = c; col_a[c] = r; return True
        return False

    def _min_line_cover(self, mat):
        n = self.n
        row_a = [-1]*n; col_a = [-1]*n
        for i in range(n):
            self._augment(i, mat, row_a, col_a, set())
        marked_rows = set(i for i in range(n) if row_a[i] == -1)
        marked_cols = set()
        changed = True
        while changed:
            changed = False
            for i in list(marked_rows):
                for j in range(n):
                    if mat[i,j] == 0 and j not in marked_cols:
                        marked_cols.add(j); changed = True
            for j in list(marked_cols):
                if col_a[j] != -1 and col_a[j] not in marked_rows:
                    marked_rows.add(col_a[j]); changed = True
        cov_rows = [i for i in range(n) if i not in marked_rows]
        cov_cols = list(marked_cols)
        return len(cov_rows)+len(cov_cols), cov_rows, cov_cols

    def _get_assignment(self, mat):
        n = self.n
        row_a = [-1]*n; col_a = [-1]*n
        for i in range(n):
            self._augment(i, mat, row_a, col_a, set())
        return row_a

    def _log(self, title, desc, mat, covered=None):
        self.steps.append({"title":title,"desc":desc,
                           "matrix":mat.copy(),"covered":covered})

    def solve(self):
        mat = self.original.copy()
        self._log("Langkah 0 — Matriks Awal",
                  f"Matriks input {self.n}×{self.n}. "
                  f"Mode: {'Minimasi' if self.mode=='min' else 'Maksimasi'}.", mat)

        if self.mode == "max":
            mv = mat.max()
            mat = mv - mat
            self._log("Langkah 1 — Konversi ke Minimasi",
                      f"Nilai maksimum = {mv:.0f}. Setiap sel = {mv:.0f} − sel_asli.", mat)
            off = 1
        else:
            off = 0

        rm = mat.min(axis=1)
        mat -= rm[:,np.newaxis]
        self._log(f"Langkah {1+off} — Row Reduction",
                  f"Min per baris: {rm.astype(int).tolist()}. "
                  "Kurangi setiap baris dengan nilai minimumnya.", mat)

        cm = mat.min(axis=0)
        mat -= cm[np.newaxis,:]
        self._log(f"Langkah {2+off} — Column Reduction",
                  f"Min per kolom: {cm.astype(int).tolist()}. "
                  "Kurangi setiap kolom dengan nilai minimumnya.", mat)

        iteration = 0
        while True:
            iteration += 1
            lines, cov_rows, cov_cols = self._min_line_cover(mat)

            if lines >= self.n:
                asgn = self._get_assignment(mat)
                self._log(
                    f"Langkah {2+off+iteration} — ✅ OPTIMAL "
                    f"(Iterasi {iteration}: {lines} = {self.n} garis)",
                    f"Jumlah garis penutup = {lines} ≥ n = {self.n}. "
                    "Algoritma berhenti. Penugasan optimal ditetapkan.", mat)
                self.assignment = asgn
                break

            uncov = [mat[i,j] for i in range(self.n) for j in range(self.n)
                     if i not in cov_rows and j not in cov_cols]
            mu = min(uncov)
            self._log(
                f"Langkah {2+off+iteration} — Iterasi {iteration}: "
                f"Garis Penutup ({lines} < {self.n})",
                f"Garis baris={cov_rows}, kolom={cov_cols}. "
                f"Min tidak tertutup = {mu:.0f}. "
                f"Sel bebas −{mu:.0f}; perpotongan +{mu:.0f}.",
                mat, covered={"rows":cov_rows,"cols":cov_cols})

            for i in range(self.n):
                for j in range(self.n):
                    ir = i in cov_rows; ic = j in cov_cols
                    if not ir and not ic: mat[i,j] -= mu
                    elif ir and ic:      mat[i,j] += mu

        self.total = sum(self.original[i, self.assignment[i]] for i in range(self.n))
        return self.assignment, self.total

# ══════════════════════════════════════════════════════════════════════
# PLOT HELPERS  (dark theme, return fig)
# ══════════════════════════════════════════════════════════════════════
MPL_STYLE = {
    "figure.facecolor":"#0d1117","axes.facecolor":"#161b26",
    "axes.edgecolor":"#2a3550","text.color":"#dde4f0",
    "axes.labelcolor":"#8b9bbf","xtick.color":"#8b9bbf",
    "ytick.color":"#8b9bbf","grid.color":"#2a3550",
    "font.family":"monospace",
}

def make_heatmap(matrix, title, services, servers, assignment=None, mode="min"):
    with plt.style.context(MPL_STYLE):
        n  = matrix.shape[0]
        fw = max(10, n*0.9); fh = max(7, n*0.75)
        fig, ax = plt.subplots(figsize=(fw, fh))
        fig.patch.set_facecolor("#0d1117")
        cmap = "RdYlGn_r" if mode=="min" else "RdYlGn"
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", alpha=0.80)
        fs = max(5, 9 - n//4)
        for i in range(n):
            for j in range(n):
                v  = matrix[i,j]; is_a = assignment is not None and assignment[i]==j
                col = "#22c55e" if is_a else ("#dde4f0" if v < matrix.max()*0.65 else "#111")
                ax.text(j, i, f"★\n{v:.0f}" if is_a else f"{v:.0f}",
                        ha="center", va="center", color=col, fontsize=fs,
                        fontweight="bold" if is_a else "normal")
                if is_a:
                    ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,
                                 lw=2.5,edgecolor="#22c55e",facecolor="none"))
        ax.set_xticks(range(n))
        ax.set_xticklabels([s[:8] for s in servers], rotation=50, ha="right",
                            fontsize=max(5,7-n//6))
        ax.set_yticks(range(n))
        ax.set_yticklabels([s[:16] for s in services], fontsize=max(5,7-n//6))
        ax.set_title(title, fontsize=12, color="#dde4f0", fontweight="bold", pad=12)
        plt.colorbar(im, ax=ax, fraction=.025, pad=.02,
                     label="Latency (ms)" if mode=="min" else "Kinerja Score")
        plt.tight_layout()
    return fig

def make_step_heatmap(step, assignment, services, servers, mode):
    with plt.style.context(MPL_STYLE):
        mat = step["matrix"]; cov = step["covered"]
        n   = mat.shape[0]
        fw  = max(9, n*0.85); fh = max(6, n*0.7)
        fig, ax = plt.subplots(figsize=(fw, fh))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b26")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a3550")
        ax.imshow(mat, cmap="RdYlGn_r" if mode=="min" else "RdYlGn",
                  aspect="auto", alpha=0.72)
        if cov:
            ov = np.zeros((n,n))
            for r in cov["rows"]: ov[r,:]=1
            for c in cov["cols"]: ov[:,c]=1
            ax.imshow(np.ma.masked_where(ov==0,ov), cmap="autumn",
                      aspect="auto", alpha=0.22)
        is_last = assignment is not None
        fs = max(5, 8 - n//4)
        for i in range(n):
            for j in range(n):
                v = mat[i,j]
                is_a = is_last and assignment[i]==j
                col  = "#22c55e" if is_a else ("#a0e0ff" if v==0 else "#dde4f0")
                ax.text(j,i,f"★{v:.0f}" if is_a else f"{v:.0f}",
                        ha="center",va="center",color=col,fontsize=fs,
                        fontweight="bold" if (is_a or v==0) else "normal")
        ax.set_xticks(range(n))
        ax.set_xticklabels([s[:5] for s in servers],
                            rotation=55, fontsize=max(4,6-n//5))
        ax.set_yticks(range(n))
        ax.set_yticklabels([s[:9] for s in services], fontsize=max(4,6-n//5))
        ax.set_title(step["title"], fontsize=8, color="#dde4f0",
                     fontweight="bold", pad=5)
        plt.tight_layout()
    return fig

def make_result_charts(original, assignment, services, servers, mode, total):
    with plt.style.context(MPL_STYLE):
        n    = len(assignment)
        vals = [original[i,assignment[i]] for i in range(n)]
        mean_v = sum(vals)/n
        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, n*0.6)))
        fig.patch.set_facecolor("#0d1117")

        # Bar chart
        ax2 = axes[0]; ax2.set_facecolor("#161b26")
        for sp in ax2.spines.values(): sp.set_edgecolor("#2a3550")
        colors = ["#22c55e" if v<=mean_v else "#f59e0b" if v<=mean_v*1.3
                  else "#ef4444" for v in vals]
        labels = [f"{services[i][:12]}\n→{servers[assignment[i]][:8]}" for i in range(n)]
        bars = ax2.barh(range(n), vals, color=colors, alpha=0.85,
                        edgecolor="#2a3550", height=0.72)
        ax2.axvline(mean_v, color="#4f8ef7", lw=1.5, ls="--",
                    label=f"Rata² {mean_v:.0f} ms")
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                     f"{v:.0f}", va="center", ha="left",
                     fontsize=max(5,7-n//5), color="#dde4f0")
        ax2.set_yticks(range(n)); ax2.set_yticklabels(labels,
                       fontsize=max(4,6-n//5), color="#8b9bbf")
        ax2.set_xlabel("Latency (ms)"); ax2.grid(axis="x", alpha=0.3)
        ax2.legend(fontsize=8, facecolor="#161b26",
                   labelcolor="#8b9bbf", edgecolor="#2a3550")
        ax2.set_title(f"Latency per Layanan  |  Total: {total:.0f} ms",
                      color="#dde4f0", fontsize=11)
        ax2.tick_params(colors="#8b9bbf")

        # Network map
        ax3 = axes[1]; ax3.set_facecolor("#161b26")
        for sp in ax3.spines.values(): sp.set_edgecolor("none")
        ax3.set_xlim(0,10); ax3.set_ylim(-0.5, n+0.5); ax3.axis("off")
        ax3.set_title("Peta Alokasi Penugasan", color="#dde4f0", fontsize=11)
        sy = n/max(n-1,1)
        for i in range(n):
            yi = (n-1-i)*sy; yj = (n-1-assignment[i])*sy
            v  = original[i,assignment[i]]
            lc = "#22c55e" if v<=mean_v else "#f59e0b" if v<=mean_v*1.3 else "#ef4444"
            ax3.annotate("", xy=(7.2,yj), xytext=(2.8,yi),
                         arrowprops=dict(arrowstyle="->",color=lc,lw=1.3,alpha=0.8))
            ax3.text(0.1,yi,services[i][:14],va="center",ha="left",fontsize=6.5,
                     color="#4f8ef7",
                     bbox=dict(boxstyle="round,pad=0.2",facecolor="#0d2040",
                               edgecolor="#4f8ef7",lw=0.8))
            ax3.text(9.9,yj,servers[assignment[i]][:10],va="center",ha="right",fontsize=6.5,
                     color="#22c55e",
                     bbox=dict(boxstyle="round,pad=0.2",facecolor="#0d2018",
                               edgecolor="#22c55e",lw=0.8))
            ax3.text(5.0,(yi+yj)/2,f"{v:.0f}",va="center",ha="center",
                     fontsize=6,color=lc,
                     bbox=dict(boxstyle="round,pad=0.1",facecolor="#161b26",
                               edgecolor=lc,lw=0.6,alpha=0.9))
        ax3.text(0.1,n*sy+0.2,"LAYANAN KAMPUS",fontsize=8,
                 color="#4f8ef7",fontweight="bold")
        ax3.text(9.9,n*sy+0.2,"KLASTER SERVER",fontsize=8,
                 color="#22c55e",fontweight="bold",ha="right")
        patches = [mpatches.Patch(color="#22c55e",label=f"≤ rata² ({mean_v:.0f})"),
                   mpatches.Patch(color="#f59e0b",label="Sedang"),
                   mpatches.Patch(color="#ef4444",label="Tinggi")]
        ax3.legend(handles=patches,loc="lower center",fontsize=7,
                   facecolor="#161b26",labelcolor="#8b9bbf",edgecolor="#2a3550",
                   ncol=3,bbox_to_anchor=(0.5,-0.03))
        plt.tight_layout()
    return fig

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#0d1117")
    buf.seek(0)
    return buf.getvalue()

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════
for key in ["matrix","result","hun"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 16px'>
      <div style='font-size:36px'>☁️</div>
      <div style='font-size:15px;font-weight:700;color:#dde4f0'>Alokasi Server Cloud</div>
      <div style='font-size:12px;color:#4f6080;margin-top:4px'>Algoritma Hungarian</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("⚙️ Konfigurasi")

    N    = st.selectbox("Ukuran Matriks (n × n)",
                        options=list(range(8,21)),
                        index=2,
                        format_func=lambda x: f"{x} × {x}")
    MODE = st.radio("Mode Optimasi",
                    ["Minimasi Latency (ms)", "Maksimasi Kinerja"],
                    index=0)
    mode = "min" if "Minimasi" in MODE else "max"

    st.divider()
    st.subheader("🔢 Pengisian Data")

    fill_mode = st.radio("Metode Pengisian",
                         ["🔀 Acak (Random)", "✏️ Manual"])

    if fill_mode == "🔀 Acak (Random)":
        col1, col2 = st.columns(2)
        rmin = col1.number_input("Min (ms)", 1, 998, 10)
        rmax = col2.number_input("Max (ms)", 2, 999, 200)
        seed = st.number_input("Seed", 0, 9999, 42)

    st.divider()
    st.markdown("""
    <div style='font-size:11px;color:#4f6080;line-height:1.8'>
    📚 Riset Operasi · SIS3532<br>
    👩‍🏫 Rillya Arundaa, S.Kom, M.Kom<br>
    👥 Kelompok 10 · UNSRAT<br>
    📌 Topik 1 · Kasus 2
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
SERVICES = ALL_SERVICES[:N]
SERVERS  = ALL_SERVERS[:N]

st.title("☁️ Alokasi Server Cloud Kampus")
st.markdown(
    f"**Topik 1 · Kasus 2 · Riset Operasi SIS3532** — "
    f"Penugasan **{N} layanan aplikasi** ke **{N} klaster server** "
    f"menggunakan **Algoritma Hungarian** untuk "
    f"{'meminimalkan total latency (ms)' if mode=='min' else 'memaksimalkan kinerja server'}."
)
st.markdown('<div class="info-box">📌 <b>Cara Pakai:</b> '
            'Atur konfigurasi di sidebar kiri → '
            'Klik <b>Generate Matriks</b> → '
            'Klik <b>▶ Jalankan Algoritma Hungarian</b> → '
            'Lihat hasil di tab Hasil & Langkah-Langkah.</div>',
            unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Input Matriks",
    "✅ Hasil Optimal",
    "📋 Langkah-Langkah",
    "📖 Teori & Panduan",
])

# ════════════════════════════ TAB 1: INPUT ════════════════════════════
with tab1:
    st.subheader(f"Matriks Latency {N}×{N}")
    st.caption("Baris = Layanan Aplikasi Kampus  |  Kolom = Klaster Server  |  Nilai = Latency (ms)")

    # Generate / reset
    col_a, col_b, col_c = st.columns([1,1,2])
    with col_a:
        if st.button("🔀 Generate Matriks", use_container_width=True):
            rng = np.random.default_rng(seed if fill_mode=="🔀 Acak (Random)" else 42)
            st.session_state.matrix = rng.integers(
                rmin if fill_mode=="🔀 Acak (Random)" else 10,
                (rmax if fill_mode=="🔀 Acak (Random)" else 200) + 1,
                size=(N,N)
            ).astype(float)
            st.session_state.result = None
            st.session_state.hun    = None
    with col_b:
        if st.button("🗑 Reset", use_container_width=True):
            st.session_state.matrix = None
            st.session_state.result = None
            st.session_state.hun    = None

    # Manual input atau show generated
    if fill_mode == "✏️ Manual":
        st.info("Isi setiap sel di bawah ini. Klik di luar sel untuk menyimpan.")
        if st.session_state.matrix is None or st.session_state.matrix.shape[0] != N:
            st.session_state.matrix = np.ones((N,N), dtype=float) * 50.0
        df_edit = pd.DataFrame(
            st.session_state.matrix,
            index=SERVICES, columns=SERVERS
        )
        edited = st.data_editor(df_edit, use_container_width=True)
        st.session_state.matrix = edited.values.astype(float)
    else:
        if st.session_state.matrix is not None and st.session_state.matrix.shape[0] == N:
            assign_show = st.session_state.result["assignment"] if st.session_state.result else None

            # Heatmap
            fig_h = make_heatmap(
                st.session_state.matrix,
                f"Matriks Latency {'+ Penugasan Optimal (★)' if assign_show else 'Awal'} — {N}×{N}",
                SERVICES, SERVERS, assign_show, mode
            )
            st.pyplot(fig_h, use_container_width=True)
            plt.close(fig_h)

            # Download heatmap
            st.download_button("⬇ Download Heatmap",
                               fig_to_bytes(make_heatmap(
                                   st.session_state.matrix,"Heatmap",
                                   SERVICES,SERVERS,assign_show,mode)),
                               "heatmap.png","image/png")

            # DataFrame
            with st.expander("📋 Lihat Nilai Tabel"):
                df_show = pd.DataFrame(
                    st.session_state.matrix.astype(int),
                    index=SERVICES, columns=SERVERS
                )
                st.dataframe(df_show, use_container_width=True)
        else:
            st.warning("Klik **🔀 Generate Matriks** terlebih dahulu.")

    # Run button
    st.divider()
    run_col, _ = st.columns([1,3])
    with run_col:
        run_btn = st.button("▶ Jalankan Algoritma Hungarian",
                            type="primary", use_container_width=True)

    if run_btn:
        if st.session_state.matrix is None or st.session_state.matrix.shape[0] != N:
            st.error("Generate atau isi matriks terlebih dahulu!")
        else:
            with st.spinner("Menjalankan algoritma..."):
                hun = HungarianAlgorithm(st.session_state.matrix, mode)
                asgn, total = hun.solve()
                vals = [st.session_state.matrix[i, asgn[i]] for i in range(N)]
                st.session_state.hun    = hun
                st.session_state.result = {
                    "assignment": asgn,
                    "total"     : total,
                    "vals"      : vals,
                    "steps"     : hun.steps,
                }
            st.success(f"✅ Selesai! Total optimal: **{total:.0f} ms** dalam {len(hun.steps)} langkah.")
            st.info("Buka tab **✅ Hasil Optimal** dan **📋 Langkah-Langkah** untuk detail.")

# ════════════════════════════ TAB 2: HASIL ════════════════════════════
with tab2:
    if st.session_state.result is None:
        st.info("Jalankan algoritma di tab **📊 Input Matriks** terlebih dahulu.")
    else:
        res  = st.session_state.result
        asgn = res["assignment"]
        tot  = res["total"]
        vals = res["vals"]

        # Metrics
        st.subheader("📈 Ringkasan Hasil")
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total Optimal",       f"{tot:.0f} ms")
        m2.metric("Rata-rata / Layanan", f"{sum(vals)/N:.1f} ms")
        m3.metric("Latency Terendah",    f"{min(vals):.0f} ms")
        m4.metric("Latency Tertinggi",   f"{max(vals):.0f} ms")
        m5.metric("Std. Deviasi",        f"{float(np.std(vals)):.1f} ms")

        st.divider()

        # Result table
        st.subheader("📋 Tabel Penugasan Optimal")
        rows = []
        for i in range(N):
            j = asgn[i]; v = st.session_state.matrix[i,j]
            rows.append({
                "No."                   : i+1,
                "Layanan Aplikasi Kampus": SERVICES[i],
                "→"                     : "→",
                "Klaster Server"        : SERVERS[j],
                "Latency (ms)"          : int(v),
                "Status"                : "✅ Optimal",
            })
        df_res = pd.DataFrame(rows).set_index("No.")
        st.dataframe(df_res, use_container_width=True)

        # CSV download
        csv_bytes = df_res.to_csv().encode()
        st.download_button("⬇ Download CSV", csv_bytes,
                           "hasil_penugasan.csv","text/csv")

        st.divider()

        # Charts
        st.subheader("📊 Visualisasi Hasil")
        fig_r = make_result_charts(st.session_state.matrix, asgn,
                                    SERVICES, SERVERS, mode, tot)
        st.pyplot(fig_r, use_container_width=True)
        st.download_button("⬇ Download Chart",
                           fig_to_bytes(fig_r),
                           "chart_hasil.png","image/png")
        plt.close(fig_r)

        # Verifikasi
        st.divider()
        st.subheader("🔍 Verifikasi")
        if N <= 9:
            with st.spinner(f"Brute-force {math.factorial(N):,} permutasi..."):
                best = float("inf") if mode=="min" else float("-inf")
                for perm in permutations(range(N)):
                    v = sum(st.session_state.matrix[i,perm[i]] for i in range(N))
                    if (mode=="min" and v<best) or (mode=="max" and v>best):
                        best = v
            ok = abs(best - tot) < 1e-6
            if ok:
                st.success(f"✅ Brute-force = {best:.0f} ms == Hungarian = {tot:.0f} ms — **VALID!**")
            else:
                st.error(f"❌ Brute-force = {best:.0f} ms ≠ Hungarian = {tot:.0f} ms")
        else:
            st.info(f"N={N} terlalu besar untuk brute-force. Hasil Hungarian valid secara matematis (O(n³)).")

# ═════════════════════════ TAB 3: LANGKAH ════════════════════════════
with tab3:
    if st.session_state.result is None:
        st.info("Jalankan algoritma di tab **📊 Input Matriks** terlebih dahulu.")
    else:
        steps = st.session_state.result["steps"]
        asgn  = st.session_state.result["assignment"]
        st.subheader(f"📋 {len(steps)} Langkah Transformasi Matriks")
        st.caption("Sel ★ hijau = penugasan optimal | Sel ◉ = nilai nol | Sel kuning = tertutup garis penutup")

        for idx, step in enumerate(steps):
            is_last = (idx == len(steps)-1)
            with st.expander(f"{'🏁' if is_last else '🔹'} {step['title']}", expanded=is_last):
                st.markdown(f'<div class="step-desc">📌 {step["desc"]}</div>',
                            unsafe_allow_html=True)
                fig_s = make_step_heatmap(
                    step,
                    asgn if is_last else None,
                    SERVICES, SERVERS, mode
                )
                st.pyplot(fig_s, use_container_width=True)

                if step["covered"]:
                    cov = step["covered"]
                    c1, c2 = st.columns(2)
                    c1.info(f"**Garis Baris:** {[SERVICES[r][:15] for r in cov['rows']]}")
                    c2.info(f"**Garis Kolom:** {[SERVERS[c] for c in cov['cols']]}")
                plt.close(fig_s)

# ══════════════════════════ TAB 4: PANDUAN ═══════════════════════════
with tab4:
    st.subheader("📖 Teori & Panduan Algoritma Hungarian")

    st.markdown("""
    ### 🎯 Tentang Studi Kasus
    **Topik 1 · Kasus 2 — Alokasi Server Cloud Kampus**

    Masalah: Mengalokasikan **n layanan aplikasi kampus** ke **n klaster server**
    secara optimal sehingga total *latency* diminimalkan (Minimasi) atau total
    kinerja dimaksimalkan (Maksimasi).

    Ini adalah **Assignment Problem** — kelas khusus Linear Programming
    yang diselesaikan secara efisien oleh **Algoritma Hungarian** (Harold Kuhn, 1955).
    """)

    with st.expander("📐 Formula & Langkah Algoritma", expanded=True):
        st.markdown("""
        **Dua fase utama:**

        **1️⃣ Reduksi Matriks**
        - **Row Reduction:** Kurangi setiap baris dengan nilai minimumnya → menjamin minimal satu nol per baris
        - **Column Reduction:** Kurangi setiap kolom dengan nilai minimumnya → menjamin minimal satu nol per kolom

        **2️⃣ Iterasi (hingga optimal)**
        - Tutup semua nol dengan **jumlah garis minimum** (Teorema König)
        - Jika jumlah garis = n → **OPTIMAL**, baca penugasan dari nol terpilih
        - Jika jumlah garis < n → cari nilai minimum sel tidak tertutup (θ):
          - Sel tidak tertutup: **−θ**
          - Sel perpotongan dua garis: **+θ**
          - Sel tertutup satu garis: **tetap**
        - Ulangi

        **Formula:**
        ```
        ES(i) tidak berlaku di sini — ini assignment problem, bukan CPM.

        Hasil:  assignment[i] = j  artinya layanan i → server j
        Total = Σ matrix[i][assignment[i]]  untuk i = 0..n-1
        ```

        **Kompleksitas:** O(n³) — jauh lebih efisien dari brute-force O(n!)
        """)

    with st.expander("⚙️ Cara Menggunakan Aplikasi"):
        st.markdown("""
        1. **Atur ukuran matriks** (8–20) di sidebar kiri
        2. **Pilih mode**: Minimasi Latency atau Maksimasi Kinerja
        3. **Pilih metode pengisian**: Acak atau Manual
        4. Klik **🔀 Generate Matriks** (mode acak) atau isi tabel (mode manual)
        5. Klik **▶ Jalankan Algoritma Hungarian**
        6. Lihat **✅ Hasil Optimal** untuk ringkasan dan visualisasi
        7. Lihat **📋 Langkah-Langkah** untuk transparansi perhitungan
        8. Download CSV atau PNG menggunakan tombol download
        """)

    with st.expander("💡 Interpretasi Hasil"):
        st.markdown("""
        | Kondisi | Arti |
        |---------|------|
        | **Total kecil (mode min)** | Alokasi sangat efisien, latency rendah |
        | **Sel ★ hijau** | Pasangan layanan–server yang terpilih optimal |
        | **Banyak iterasi** | Matriks lebih kompleks, butuh lebih banyak penyesuaian |
        | **Slack tinggi** | Tidak berlaku di Hungarian; semua penugasan adalah 1-1 |
        | **Verifikasi OK** | Brute-force mengkonfirmasi hasil Hungarian identik |

        **Rekomendasi Manajerial:**
        - Layanan dengan latency tinggi → pertimbangkan upgrade klaster yang ditugaskan
        - Hasil ini optimal secara global; tidak ada penugasan lain yang menghasilkan total lebih kecil
        """)

    st.divider()
    st.markdown("""
    <div style='text-align:center;color:#4f6080;font-size:12px;padding:10px'>
    Riset Operasi SIS3532 · Kelompok 10 · UNSRAT ·
    Dosen: Rillya Arundaa, S.Kom, M.Kom
    </div>
    """, unsafe_allow_html=True)
