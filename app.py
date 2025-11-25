import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings

# Ignora warnings irrelevantes
warnings.filterwarnings("ignore")


# ==============================================================================
# 1. CLASSES DE CÁLCULO (Robustas e Fisicamente Corretas)
# ==============================================================================

class MinimumCurvature:
    """Calculo de trajetoria 3D usando Minimum Curvature Method (Padrao SPE)"""
    
    @staticmethod
    def calculate_survey(md_array, inc_array, azi_array, surface_coords=(0, 0, 0)):
        """
        Calcula coordenadas 3D (N, E, TVD) usando Minimum Curvature.
        """
        n_points = len(md_array)
        
        # Inicializa arrays
        tvd = np.zeros(n_points)
        north = np.zeros(n_points)
        east = np.zeros(n_points)
        dls = np.zeros(n_points)
        
        # Coordenadas iniciais
        north[0], east[0], tvd[0] = surface_coords
        
        # Converte para radianos
        inc_rad = np.radians(inc_array)
        azi_rad = np.radians(azi_array)
        
        for i in range(1, n_points):
            # Diferenca de MD
            dmd = md_array[i] - md_array[i-1]
            
            # Evita divisao por zero
            if dmd <= 1e-6:
                north[i] = north[i-1]
                east[i] = east[i-1]
                tvd[i] = tvd[i-1]
                dls[i] = 0
                continue
            
            # Angulos do intervalo
            i1, i2 = inc_rad[i-1], inc_rad[i]
            a1, a2 = azi_rad[i-1], azi_rad[i]
            
            # Dog-leg angle (beta)
            cos_beta = (np.cos(i1) * np.cos(i2) + 
                       np.sin(i1) * np.sin(i2) * np.cos(a2 - a1))
            cos_beta = np.clip(cos_beta, -1, 1)
            beta = np.arccos(cos_beta)
            
            # DLS em graus/30m
            dls[i] = np.degrees(beta) / dmd * 30.0
            
            # Ratio Factor (RF)
            if beta < 1e-6:
                rf = 1.0
            else:
                rf = (2 / beta) * np.tan(beta / 2)
            
            # Calculos de deslocamento
            delta_tvd = 0.5 * dmd * (np.cos(i1) + np.cos(i2)) * rf
            delta_north = 0.5 * dmd * (np.sin(i1) * np.cos(a1) + 
                                       np.sin(i2) * np.cos(a2)) * rf
            delta_east = 0.5 * dmd * (np.sin(i1) * np.sin(a1) + 
                                      np.sin(i2) * np.sin(a2)) * rf
            
            # Acumula coordenadas
            tvd[i] = tvd[i-1] + delta_tvd
            north[i] = north[i-1] + delta_north
            east[i] = east[i-1] + delta_east
        
        return pd.DataFrame({
            'MD': md_array,
            'Inc': inc_array,
            'Azi': azi_array,
            'TVD': tvd,
            'N': north,
            'E': east,
            'DLS': dls
        })


class SingleTrajectoryTortuosity:
    def __init__(self, md_spacing, smoothing_window, mwd_noise_mode, mwd_noise_factor):
        self.md_spacing = md_spacing
        self.smoothing_window = smoothing_window
        self.mwd_noise_mode = mwd_noise_mode
        self.mwd_noise_factor = mwd_noise_factor
        
        # Parametros base ajustados (Padrão ISCWSA Rev5 para MWD Magnético)
        self.sigma_inc_base = 0.15 
        self.sigma_azi_base = 0.40
        
        self.deviation_model = None
        self.knn_model = None
        self.feature_scaler = StandardScaler()
        self.training_features = None
    
    def _get_dynamic_uncertainty(self, inc_array):
        """
        Estimativa física realista de incerteza MWD.
        """
        inc_rad = np.radians(np.clip(inc_array, 0, 180))
        
        # Formula: Base / (1 + sin(Inc)) -> Pior na vertical
        sigma_inc = self.sigma_inc_base / (1.0 + np.sin(inc_rad))
        
        # Formula: Base * (1 + 0.5 * |cos(Inc)|) -> Pior na vertical e horizontal
        sigma_azi = self.sigma_azi_base * (1.0 + 0.5 * np.abs(np.cos(inc_rad)))
        
        return sigma_inc, sigma_azi
    
    def _interpolate_azimuth_slerp(self, md_orig, azi_orig, md_new):
        """Interpolacao esferica manual para azimute"""
        azi_rad = np.radians(azi_orig)
        x = np.cos(azi_rad)
        y = np.sin(azi_rad)
        
        x_interp = np.interp(md_new, md_orig, x)
        y_interp = np.interp(md_new, md_orig, y)
        
        return np.degrees(np.arctan2(y_interp, x_interp)) % 360
    
    def load_trajectory_from_df(self, df):
        """Carrega, limpa e uniformiza trajetoria"""
        for col in ["MD", "Inc", "Azi"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna(subset=["MD", "Inc", "Azi"]).reset_index(drop=True)
        
        # SEGURANÇA: Ordena e remove duplicatas de MD
        df = df.sort_values("MD").drop_duplicates("MD")
        
        if len(df) < 2:
            raise ValueError("Trajetoria precisa ter pelo menos 2 pontos validos")
        
        # Grid uniforme
        md_uniform = np.arange(df["MD"].min(), df["MD"].max() + self.md_spacing, 
                               self.md_spacing)
        
        # Interpolacao
        inc_interp = interp1d(df["MD"], df["Inc"], kind="linear", 
                             fill_value="extrapolate", bounds_error=False)
        
        azi_uniform = self._interpolate_azimuth_slerp(df["MD"].values, 
                                                       df["Azi"].values, 
                                                       md_uniform)
        
        # Se DF original tem coords de origem, usa
        origin_n = df["N"].iloc[0] if "N" in df.columns else 0
        origin_e = df["E"].iloc[0] if "E" in df.columns else 0
        origin_tvd = df["TVD"].iloc[0] if "TVD" in df.columns else 0
        
        # Calcula Survey
        survey = MinimumCurvature.calculate_survey(
            md_uniform,
            np.clip(inc_interp(md_uniform), 0, 180),
            azi_uniform,
            surface_coords=(origin_n, origin_e, origin_tvd)
        )
        
        return survey
    
    def build_deviation_model(self, planned_dfs, executed_dfs):
        """Treina modelo KNN com dados historicos"""
        all_deviations = []
        well_stats = {}
        
        for idx, (plan, exec_) in enumerate(zip(planned_dfs, executed_dfs)):
            try:
                plan_uniform = self.load_trajectory_from_df(plan)
                exec_uniform = self.load_trajectory_from_df(exec_)
                
                # Merge por MD
                merged = pd.merge_asof(
                    plan_uniform, exec_uniform, 
                    on="MD", suffixes=("_plan", "_exec"),
                    direction="nearest", tolerance=self.md_spacing * 2
                ).dropna()
                
                if len(merged) < 5:
                    continue
                
                # Deltas
                merged["delta_inc"] = merged["Inc_exec"] - merged["Inc_plan"]
                delta_azi = merged["Azi_exec"] - merged["Azi_plan"]
                merged["delta_azi"] = ((delta_azi + 180) % 360) - 180
                
                # Deadband
                if self.mwd_noise_mode == "subtract":
                    sig_inc, sig_azi = self._get_dynamic_uncertainty(merged["Inc_plan"].values)
                    
                    merged["delta_inc"] = np.sign(merged["delta_inc"]) * np.maximum(
                        0, np.abs(merged["delta_inc"]) - (sig_inc * self.mwd_noise_factor)
                    )
                    merged["delta_azi"] = np.sign(merged["delta_azi"]) * np.maximum(
                        0, np.abs(merged["delta_azi"]) - (sig_azi * self.mwd_noise_factor)
                    )
                
                # Suavizacao historica
                if self.smoothing_window > 0:
                    merged["delta_inc"] = gaussian_filter1d(merged["delta_inc"], sigma=self.smoothing_window)
                    merged["delta_azi"] = gaussian_filter1d(merged["delta_azi"], sigma=self.smoothing_window)
                
                # Features
                md_range = merged["MD"].max() - merged["MD"].min()
                merged["MD_norm"] = ((merged["MD"] - merged["MD"].min()) / md_range) if md_range > 0 else 0
                merged["section"] = self._classify_section(merged["Inc_plan"].values)
                
                deviations = merged[["MD_norm", "Inc_plan", "Azi_plan", "delta_inc", "delta_azi", "section"]].copy()
                deviations.columns = ["MD_norm", "Inc", "Azi", "delta_inc", "delta_azi", "section"]
                all_deviations.append(deviations)
                
                well_stats[f"Poco_{idx+1}"] = {
                    "pontos": len(deviations),
                    "delta_inc_mean": deviations["delta_inc"].mean(),
                    "delta_inc_std": deviations["delta_inc"].std(),
                    "delta_azi_mean": deviations["delta_azi"].mean(),
                    "delta_azi_std": deviations["delta_azi"].std(),
                    "section_dist": deviations["section"].value_counts().to_dict()
                }
            except Exception:
                continue
        
        if not all_deviations:
            raise ValueError("Nenhum dado valido extraido dos pocos de correlacao")
        
        self.deviation_model = pd.concat(all_deviations, ignore_index=True)
        
        # Treino KNN
        features = self.deviation_model[["MD_norm", "Inc"]].values
        self.training_features = self.feature_scaler.fit_transform(features)
        
        # SEGURANÇA: KNN k <= n_samples
        n_samples = len(self.deviation_model)
        k_calculated = min(20, max(5, n_samples // 10))
        k_neighbors = min(k_calculated, n_samples)
        
        if k_neighbors < 1:
            raise ValueError("Dados insuficientes para treinamento do modelo.")

        self.knn_model = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
        self.knn_model.fit(self.training_features)
        
        return self.deviation_model, well_stats
    
    def _classify_section(self, inc_array):
        """Classifica secao do poco"""
        conditions = [inc_array < 3, (inc_array >= 3) & (inc_array < 30), 
                     (inc_array >= 30) & (inc_array < 60), inc_array >= 60]
        choices = ["vertical", "buildup", "tangent", "horizontal"]
        return np.select(conditions, choices, default="tangent")

    def apply_tortuosity(self, planned_df, max_dls=10.0, smoothing_passes=2):
        if self.knn_model is None:
            raise ValueError("Modelo nao treinado.")
        
        # 1. Carrega Planejado Original
        plan = self.load_trajectory_from_df(planned_df)
        
        # 2. Salva Coordenadas de Referência (TARGET) para cálculo de Drift correto
        ref_n = plan["N"].iloc[-1]
        ref_e = plan["E"].iloc[-1]
        ref_tvd = plan["TVD"].iloc[-1]
        
        # Features
        md_range = plan["MD"].max() - plan["MD"].min()
        plan["MD_norm"] = (plan["MD"] - plan["MD"].min()) / md_range if md_range > 1e-3 else 0
        
        # 3. Inferencia KNN
        query_features = self.feature_scaler.transform(plan[["MD_norm", "Inc"]].values)
        distances, indices = self.knn_model.kneighbors(query_features)
        
        weights = 1.0 / np.maximum(distances, 1e-8)
        sum_weights = np.sum(weights, axis=1, keepdims=True)
        
        neighbor_delta_inc = self.deviation_model.iloc[indices.flatten()]["delta_inc"].values.reshape(indices.shape)
        neighbor_delta_azi = self.deviation_model.iloc[indices.flatten()]["delta_azi"].values.reshape(indices.shape)
        
        plan["delta_inc"] = np.sum(neighbor_delta_inc * weights, axis=1) / sum_weights.flatten()
        plan["delta_azi"] = np.sum(neighbor_delta_azi * weights, axis=1) / sum_weights.flatten()
        
        # 4. Adicao de Ruido
        if self.mwd_noise_mode == "add":
            sig_inc, sig_azi = self._get_dynamic_uncertainty(plan["Inc"].values)
            std_local_inc = np.std(neighbor_delta_inc, axis=1)
            std_local_azi = np.std(neighbor_delta_azi, axis=1)
            
            noise_scale_inc = np.sqrt(std_local_inc**2 + (sig_inc * self.mwd_noise_factor)**2)
            noise_scale_azi = np.sqrt(std_local_azi**2 + (sig_azi * self.mwd_noise_factor)**2)
            
            plan["delta_inc"] += np.random.normal(0, noise_scale_inc * 0.3, size=len(plan))
            plan["delta_azi"] += np.random.normal(0, noise_scale_azi * 0.3, size=len(plan))
        
        # 5. Aplica Deltas
        plan["Inc_adjusted"] = np.clip(plan["Inc"] + plan["delta_inc"], 0, 120)
        plan["Azi_adjusted"] = (plan["Azi"] + plan["delta_azi"]) % 360
        
        # 6. Suavizacao
        for _ in range(smoothing_passes):
            plan["Inc_adjusted"] = gaussian_filter1d(plan["Inc_adjusted"], sigma=1.5)
            
            # Suavizacao Vetorial
            azi_rad = np.radians(plan["Azi_adjusted"])
            sin_az = np.sin(azi_rad)
            cos_az = np.cos(azi_rad)
            
            sin_smooth = gaussian_filter1d(sin_az, sigma=1.5)
            cos_smooth = gaussian_filter1d(cos_az, sigma=1.5)
            
            # Renormaliza vetor
            norm = np.sqrt(sin_smooth**2 + cos_smooth**2)
            sin_smooth /= np.maximum(norm, 1e-8)
            cos_smooth /= np.maximum(norm, 1e-8)
            
            plan["Azi_adjusted"] = np.degrees(np.arctan2(sin_smooth, cos_smooth)) % 360
        
        plan["Inc_adjusted"] = np.clip(plan["Inc_adjusted"], 0, 90)
        
        # 7. Limitador de DLS (Forward)
        inc_adj = plan["Inc_adjusted"].values.copy()
        azi_adj = plan["Azi_adjusted"].values.copy()
        md = plan["MD"].values
        dls_violations = 0
        
        for i in range(1, len(plan)):
            dmd = md[i] - md[i-1]
            if dmd <= 1e-6: continue
            
            i1, i2 = np.radians(inc_adj[i-1]), np.radians(inc_adj[i])
            a1, a2 = np.radians(azi_adj[i-1]), np.radians(azi_adj[i])
            
            cos_dls = np.clip(np.cos(i1)*np.cos(i2) + np.sin(i1)*np.sin(i2)*np.cos(a2-a1), -1, 1)
            dls_val = np.degrees(np.arccos(cos_dls)) / dmd * 30.0
            
            if dls_val > max_dls:
                ratio = max_dls / dls_val
                inc_adj[i] = inc_adj[i-1] + (inc_adj[i] - inc_adj[i-1]) * ratio
                
                diff_azi = ((azi_adj[i] - azi_adj[i-1] + 180) % 360) - 180
                azi_adj[i] = (azi_adj[i-1] + diff_azi * ratio) % 360
                dls_violations += 1
        
        plan["Inc_adjusted"] = inc_adj
        plan["Azi_adjusted"] = azi_adj
        
        # 8. Calculo Final de Coordenadas
        survey_final = MinimumCurvature.calculate_survey(
            plan["MD"].values,
            plan["Inc_adjusted"].values,
            plan["Azi_adjusted"].values,
            surface_coords=(plan["N"].iloc[0], plan["E"].iloc[0], plan["TVD"].iloc[0])
        )
        
        # 9. Calcula Drift Real
        drift_n = survey_final["N"].iloc[-1] - ref_n
        drift_e = survey_final["E"].iloc[-1] - ref_e
        drift_tvd = survey_final["TVD"].iloc[-1] - ref_tvd
        
        survey_final.attrs["drift_tvd"] = abs(drift_tvd)
        survey_final.attrs["drift_horizontal"] = np.sqrt(drift_n**2 + drift_e**2)
        survey_final.attrs["drift_north"] = abs(drift_n)
        survey_final.attrs["drift_east"] = abs(drift_e)
        survey_final.attrs["dls_violations"] = dls_violations
        survey_final.attrs["dls_violation_rate"] = dls_violations / len(plan) * 100
        
        return survey_final


# ==============================================================================
# 2. FUNÇÕES AUXILIARES DE PARSER E EXPORT (Seguras)
# ==============================================================================

def parse_trajectory_file(uploaded_file):
    """Parser seguro com deteccao inteligente e uso de buffer"""
    trajectories = []
    
    try:
        file_name = str(uploaded_file.name)
        # SEGURANÇA: Lê para buffer para evitar problemas de ponteiro
        file_buffer = io.BytesIO(uploaded_file.getvalue())
        
        if file_name.endswith(".csv"):
            content = file_buffer.read().decode("utf-8")
            lines = content.split("\n")
            if not lines: return []

            # Detecta header
            header_row = 0
            for i, line in enumerate(lines[:20]):
                line_lower = line.lower()
                if any(x in line_lower for x in ["md", "depth", "inc", "azi"]):
                    header_row = i
                    break
            
            try:
                sep = ";" if ";" in lines[header_row] else ","
                # Tenta forçar decimal com virgula se for ;
                decimal = "," if sep == ";" else "."
                df = pd.read_csv(io.StringIO(content), sep=sep, decimal=decimal, skiprows=header_row)
            except:
                df = pd.read_csv(io.StringIO(content), skiprows=header_row)
                
            df = _standardize_columns(df)
            if _validate_trajectory(df):
                trajectories.append((file_name, df))
        
        elif file_name.endswith((".xlsx", ".xls")):
            excel_file = pd.ExcelFile(file_buffer)
            for sheet_name in excel_file.sheet_names:
                for skip in [0, 1, 2, 3]:
                    try:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=skip)
                        df = _standardize_columns(df)
                        if _validate_trajectory(df):
                            trajectories.append((str(sheet_name), df))
                            break
                    except:
                        continue

    except Exception as e:
        st.error(f"Erro ao processar {uploaded_file.name}: {str(e)}")
    
    return trajectories


def _standardize_columns(df):
    cols_map = {}
    for col in df.columns:
        c = str(col).lower().strip()
        if any(x in c for x in ["md", "depth", "measured"]): cols_map[col] = "MD"
        elif "inc" in c: cols_map[col] = "Inc"
        elif any(x in c for x in ["azi", "azm"]): cols_map[col] = "Azi"
        elif "tvd" in c: cols_map[col] = "TVD"
        elif "dls" in c: cols_map[col] = "DLS"
        elif "n" == c or "north" in c: cols_map[col] = "N"
        elif "e" == c or "east" in c: cols_map[col] = "E"
    
    df = df.rename(columns=cols_map)

    # --- SEGURANÇA NUMÉRICA ---
    numeric_cols = ["MD", "Inc", "Azi", "TVD", "N", "E", "DLS"]
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def _validate_trajectory(df):
    return {"MD", "Inc", "Azi"}.issubset(df.columns) and len(df) >= 2


def export_to_csv(df):
    output = io.StringIO()
    output.write("Seq;Measured;Incl;Azimuth;TVD;Northing;Easting;DLS\n")
    output.write("#;m;deg;deg;m;m;m;deg/30m\n")
    output.write("===;========;======;=======;======;========;========;========\n")
    for i, row in df.iterrows():
        line = (f"{i+1};{row['MD']:.2f};{row['Inc']:.2f};{row['Azi']:.2f};"
               f"{row['TVD']:.2f};{row['N']:.2f};{row['E']:.2f};{row['DLS']:.2f}\n")
        output.write(line.replace(".", ","))
    return output.getvalue()


# ==============================================================================
# 3. INTERFACE COMPLETA (MAIN)
# ==============================================================================

def main():
    st.set_page_config(page_title="Gerador de Tortuosidade - Versão Completa", layout="wide")
    
    st.title("🛢️ Gerador de Tortuosidade para Simulação de Desgaste")
    st.caption("Versão Profissional (Validada) com Análise Completa de Drift")
    
    with st.expander("📘 Metodologia Implementada (Verificada)", expanded=False):
        st.markdown(r"""
        ### Implementação Robusta
        
        **1. Interpolação Segura de Ângulos**
        - Inclinação: Linear
        - Azimute: SLERP esférica com normalização de vetores
        
        **2. Cálculo 3D e Drift**
        - Método: **Minimum Curvature** (SPE)
        - Drift: Calculado comparando a posição final da trajetória gerada vs. planejada original. Não há distorção geométrica forçada.
        
        **3. Modelo de Incerteza Dinâmica (ISCWSA Rev5)**
        - Inclinação: $\sigma \propto 1/(1+\sin(I))$ (Pior na vertical)
        - Azimute: $\sigma \propto 1 + 0.5|\cos(I)|$ (Pior na vertical e horizontal)
        
        **4. Aprendizado de Máquina (KNN)**
        - Busca padrões de desvio em históricos usando MD normalizado e Inclinação.
        """)
    
    st.divider()
    
    # === SIDEBAR ===
    st.sidebar.header("⚙️ Parâmetros")
    
    md_spacing = st.sidebar.number_input("Espaçamento MD (m)", 1.0, 30.0, 10.0)
    smoothing_window = st.sidebar.slider("Suavização Histórico (σ)", 0, 10, 3)
    max_dls = st.sidebar.number_input("DLS Máximo (°/30m)", 3.0, 20.0, 10.0)
    smoothing_passes = st.sidebar.slider("Passes Suavização Final", 0, 5, 2)
    
    st.sidebar.divider()
    st.sidebar.subheader("🔬 Incerteza MWD")
    
    mwd_mode = st.sidebar.radio(
        "Modo", ["subtract", "add"], index=1,
        format_func=lambda x: "🔵 Subtrair (Otimista)" if x == "subtract" else "🔴 Adicionar (Conservador)"
    )
    
    mwd_factor = st.sidebar.slider("Fator (%)", 0, 100, 50) / 100.0
    
    # === UPLOAD ===
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📂 Planejadas (Histórico)")
        f_planned = st.file_uploader("Upload CSV/Excel", accept_multiple_files=True, key="p")
    with col2:
        st.subheader("📂 Executadas (Histórico)")
        f_executed = st.file_uploader("Upload CSV/Excel", accept_multiple_files=True, key="e")
    
    st.divider()
    st.subheader("🎯 Trajetória Alvo (Novo Poço)")
    f_target = st.file_uploader("Upload CSV/Excel", key="t")
    
    # === PROCESSAMENTO ===
    if st.button("🚀 Processar Trajetória", type="primary", width='stretch'):
        if not (f_planned and f_executed and f_target):
            st.error("❌ Faltam arquivos necessários.")
            st.stop()
        
        try:
            # 1. Parsing
            with st.spinner("Lendo arquivos..."):
                planned_trajs = []
                for f in f_planned: planned_trajs.extend([t[1] for t in parse_trajectory_file(f)])
                
                executed_trajs = []
                for f in f_executed: executed_trajs.extend([t[1] for t in parse_trajectory_file(f)])
                
                target_trajs = parse_trajectory_file(f_target)
                if not target_trajs: 
                    st.error("Arquivo alvo inválido.")
                    st.stop()
                target_df = target_trajs[0][1]
            
            if not planned_trajs or not executed_trajs:
                st.error("Não foi possível ler dados históricos válidos.")
                st.stop()
                
            # 2. Modelagem
            model = SingleTrajectoryTortuosity(md_spacing, smoothing_window, mwd_mode, mwd_factor)
            
            with st.spinner("Treinando modelo KNN..."):
                deviation_model, well_stats = model.build_deviation_model(planned_trajs, executed_trajs)
                
            with st.spinner("Gerando trajetória..."):
                result = model.apply_tortuosity(target_df, max_dls, smoothing_passes)
            
            st.success("✅ Processamento concluído!")
            st.divider()
            
            # === DASHBOARD ===
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Análise Principal",
                "📉 Estatísticas do Modelo",
                "⚠️ Análise de Drift",
                "💾 Exportar Dados"
            ])
            
            # --- TAB 1: Metrics ---
            with tab1:
                st.subheader("Métricas de Tortuosidade")
                c1, c2, c3, c4 = st.columns(4)
                
                dls_orig = target_df["DLS"].mean() if "DLS" in target_df.columns else 0
                dls_adj = result["DLS"].mean()
                dls_max = result["DLS"].max()
                viol_rate = result.attrs["dls_violation_rate"]
                
                c1.metric("DLS Médio Original", f"{dls_orig:.2f}")
                c2.metric("DLS Médio Ajustado", f"{dls_adj:.2f}", delta=f"{(dls_adj/dls_orig -1)*100:.1f}%" if dls_orig>0 else None)
                c3.metric("DLS Máximo", f"{dls_max:.2f}", delta_color="inverse", delta=f"{dls_max-max_dls:.2f}" if dls_max > max_dls else None)
                c4.metric("Correções DLS", f"{viol_rate:.1f}%")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Perfil DLS")
                    chart = result[["MD", "DLS"]].copy()
                    chart["Limite"] = max_dls
                    st.line_chart(chart.set_index("MD"))
                with col_b:
                    st.subheader("Distribuição Inc")
                    st.bar_chart(result["Inc"].value_counts(bins=10).sort_index())
                    
                st.dataframe(result.head(100), width='stretch')
                
            # --- TAB 2: Stats ---
            with tab2:
                st.subheader("Dados Históricos")
                
                # --- CORREÇÃO DO FORMAT STRING ERROR ---
                stats_df = pd.DataFrame(well_stats).T
                # A coluna 'section_dist' contém dicionários, o que causa erro no style.format
                # Removemos essa coluna da exibição aqui, pois ela é detalhada abaixo
                if "section_dist" in stats_df.columns:
                    stats_df_display = stats_df.drop(columns=["section_dist"])
                else:
                    stats_df_display = stats_df
                
                st.dataframe(stats_df_display.style.format("{:.3f}"))
                
                st.subheader("Desvios por Seção")
                section_stats = deviation_model.groupby("section").agg({
                    "delta_inc": ["count", "mean", "std"],
                    "delta_azi": ["mean", "std"]
                }).round(3)
                st.dataframe(section_stats, width='stretch')
                
            # --- TAB 3: Drift ---
            with tab3:
                st.subheader("Deslocamento Final (Gerado vs Planejado)")
                drift_tvd = result.attrs["drift_tvd"]
                drift_horiz = result.attrs["drift_horizontal"]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Drift TVD", f"{drift_tvd:.2f} m", delta_color="inverse", delta=f"{drift_tvd:.2f}" if drift_tvd > 5 else None)
                c2.metric("Drift Horizontal", f"{drift_horiz:.2f} m", delta_color="inverse", delta=f"{drift_horiz:.2f}" if drift_horiz > 10 else None)
                c3.metric("Drift N/S", f"{result.attrs['drift_north']:.2f} m")
                c4.metric("Drift E/W", f"{result.attrs['drift_east']:.2f} m")
                
                if drift_tvd > 10 or drift_horiz > 20:
                    st.error("🔴 Drift Alto Detectado: O poço simulado desviou significativamente do alvo. Revise o fator MWD ou os dados históricos.")
                
            # --- TAB 4: Export ---
            with tab4:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### CSV")
                    st.download_button("Download CSV", export_to_csv(result), "tortuosidade.csv", "text/csv", width='stretch')
                
                with col2:
                    st.markdown("#### Excel Completo")
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        result.to_excel(writer, sheet_name="Trajetoria", index=False)
                        deviation_model.to_excel(writer, sheet_name="Modelo", index=False)
                        
                        # Remove a coluna de dicionario antes de salvar para evitar problemas no Excel
                        stats_for_excel = pd.DataFrame(well_stats).T
                        if "section_dist" in stats_for_excel.columns:
                            stats_for_excel = stats_for_excel.drop(columns=["section_dist"])
                        stats_for_excel.to_excel(writer, sheet_name="Stats_Pocos")
                        
                        summary = pd.DataFrame({
                            "Parametro": ["MD Spacing", "DLS Max", "MWD Mode", "MWD Factor", "Drift TVD", "Drift Horiz"],
                            "Valor": [md_spacing, max_dls, mwd_mode, mwd_factor, drift_tvd, drift_horiz]
                        })
                        summary.to_excel(writer, sheet_name="Resumo", index=False)
                        
                    st.download_button("Download Excel", buffer.getvalue(), "analise_tortuosidade.xlsx", 
                                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        except Exception as e:
            st.error(f"Erro Crítico: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()