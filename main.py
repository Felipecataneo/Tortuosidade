import streamlit as st
import pandas as pd
import numpy as np
import io
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class MinimumCurvature:
    """Calculo de trajetoria 3D usando Minimum Curvature Method (industria padrao)"""
    
    @staticmethod
    def calculate_survey(md_array, inc_array, azi_array, surface_coords=(0, 0, 0)):
        """
        Calcula coordenadas 3D (N, E, TVD) usando Minimum Curvature.
        
        Args:
            md_array: Measured Depth em metros
            inc_array: Inclinacao em graus
            azi_array: Azimute em graus
            surface_coords: (N, E, TVD) inicial
            
        Returns:
            DataFrame com colunas [MD, Inc, Azi, TVD, N, E, DLS]
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
            
            # Ratio Factor (RF) para Minimum Curvature
            if beta < 1e-6:
                rf = 1.0  # Linha reta
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
        
        # Parametros ISCWSA para incerteza MWD
        self.sigma_inc_base = 0.15
        self.sigma_azi_base = 0.40
        self.k_factor = 0.06
        
        self.deviation_model = None
        self.knn_model = None
        self.feature_scaler = StandardScaler()
        self.training_features = None
    
    def _get_dynamic_uncertainty(self, inc_array):
        """
        Incerteza dependente da inclinacao (ISCWSA MWD Rev5).
        sigma = sigma_base * (1 + k * sin(Inc))
        """
        inc_rad = np.radians(np.clip(inc_array, 0, 180))
        factor = 1.0 + self.k_factor * np.sin(inc_rad)
        
        sigma_inc = self.sigma_inc_base * factor
        sigma_azi = self.sigma_azi_base * factor
        return sigma_inc, sigma_azi
    
    def _interpolate_azimuth_slerp(self, md_orig, azi_orig, md_new):
        """
        Interpolacao esferica de azimute usando SLERP manual.
        Trata descontinuidade 0/360 corretamente.
        """
        # Converte azimute para vetor unitario 2D
        azi_rad = np.radians(azi_orig)
        x = np.cos(azi_rad)
        y = np.sin(azi_rad)
        
        # Interpola componentes
        x_interp = np.interp(md_new, md_orig, x)
        y_interp = np.interp(md_new, md_orig, y)
        
        # Reconstroi azimute normalizado
        azi_interp = np.degrees(np.arctan2(y_interp, x_interp)) % 360
        
        return azi_interp
    
    def load_trajectory_from_df(self, df):
        """Carrega e uniformiza trajetoria com interpolacao adequada"""
        for col in ["MD", "Inc", "Azi"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna(subset=["MD", "Inc", "Azi"]).reset_index(drop=True)
        
        if len(df) < 2:
            raise ValueError("Trajetoria precisa ter pelo menos 2 pontos validos")
        
        # Grid uniforme
        md_uniform = np.arange(df["MD"].min(), df["MD"].max() + self.md_spacing, 
                               self.md_spacing)
        
        # Interpolacao LINEAR para angulos (evita Runge phenomenon)
        inc_interp = interp1d(df["MD"], df["Inc"], kind="linear", 
                             fill_value="extrapolate", bounds_error=False)
        
        # Interpolacao esferica para azimute
        azi_uniform = self._interpolate_azimuth_slerp(df["MD"].values, 
                                                       df["Azi"].values, 
                                                       md_uniform)
        
        # Monta DataFrame uniformizado
        df_uniform = pd.DataFrame({
            "MD": md_uniform,
            "Inc": np.clip(inc_interp(md_uniform), 0, 180),
            "Azi": azi_uniform
        })
        
        # Calcula coordenadas 3D com Minimum Curvature
        survey = MinimumCurvature.calculate_survey(
            df_uniform["MD"].values,
            df_uniform["Inc"].values,
            df_uniform["Azi"].values
        )
        
        return survey
    
    def build_deviation_model(self, planned_dfs, executed_dfs):
        """Constroi modelo de desvios a partir de pocos historicos"""
        all_deviations = []
        well_stats = {}
        
        for idx, (plan, exec_) in enumerate(zip(planned_dfs, executed_dfs)):
            plan_uniform = self.load_trajectory_from_df(plan)
            exec_uniform = self.load_trajectory_from_df(exec_)
            
            # Merge por MD mais proximo
            merged = pd.merge_asof(
                plan_uniform, exec_uniform, 
                on="MD", 
                suffixes=("_plan", "_exec"),
                direction="nearest", 
                tolerance=self.md_spacing * 2
            )
            merged = merged.dropna()
            
            if len(merged) < 5:
                st.warning(f"Poco {idx+1}: Poucos pontos coincidentes ({len(merged)}). Verifique compatibilidade.")
                continue
            
            # Calcula deltas
            merged["delta_inc"] = merged["Inc_exec"] - merged["Inc_plan"]
            
            # Delta azimute com wrapping correto (-180 a +180)
            delta_azi = merged["Azi_exec"] - merged["Azi_plan"]
            merged["delta_azi"] = ((delta_azi + 180) % 360) - 180
            
            # Incerteza dinamica para este poco
            sig_inc, sig_azi = self._get_dynamic_uncertainty(merged["Inc_plan"].values)
            
            # Deadband (subtracao de ruido MWD)
            if self.mwd_noise_mode == "subtract":
                threshold_inc = sig_inc * self.mwd_noise_factor
                threshold_azi = sig_azi * self.mwd_noise_factor
                
                merged["delta_inc"] = np.sign(merged["delta_inc"]) * np.maximum(
                    0, np.abs(merged["delta_inc"]) - threshold_inc
                )
                merged["delta_azi"] = np.sign(merged["delta_azi"]) * np.maximum(
                    0, np.abs(merged["delta_azi"]) - threshold_azi
                )
            
            # Suavizacao dos deltas historicos
            if self.smoothing_window > 0:
                merged["delta_inc"] = gaussian_filter1d(merged["delta_inc"], 
                                                        sigma=self.smoothing_window)
                merged["delta_azi"] = gaussian_filter1d(merged["delta_azi"], 
                                                        sigma=self.smoothing_window)
            
            # Normalizacao de MD (0 a 1)
            md_range = merged["MD"].max() - merged["MD"].min()
            merged["MD_norm"] = ((merged["MD"] - merged["MD"].min()) / md_range 
                                if md_range > 1e-3 else 0)
            
            # Classificacao de secao
            merged["section"] = self._classify_section(merged["Inc_plan"].values)
            
            # Extrai features relevantes
            deviations = merged[["MD_norm", "Inc_plan", "Azi_plan", 
                                "delta_inc", "delta_azi", "section"]].copy()
            deviations.columns = ["MD_norm", "Inc", "Azi", "delta_inc", "delta_azi", "section"]
            
            all_deviations.append(deviations)
            
            # Estatisticas por poco
            well_stats[f"Poco_{idx+1}"] = {
                "pontos": len(deviations),
                "delta_inc_mean": deviations["delta_inc"].mean(),
                "delta_inc_std": deviations["delta_inc"].std(),
                "delta_azi_mean": deviations["delta_azi"].mean(),
                "delta_azi_std": deviations["delta_azi"].std(),
                "section_dist": deviations["section"].value_counts().to_dict()
            }
        
        if not all_deviations:
            raise ValueError("Nenhum dado valido extraido dos pocos de correlacao")
        
        self.deviation_model = pd.concat(all_deviations, ignore_index=True)
        
        # === TREINO DO MODELO KNN ===
        # Features: MD_norm e Inc (ambos escalonados)
        features = self.deviation_model[["MD_norm", "Inc"]].values
        self.training_features = self.feature_scaler.fit_transform(features)
        
        # KNN com k adaptativo (minimo 5, maximo 20)
        k_neighbors = min(20, max(5, len(self.deviation_model) // 10))
        self.knn_model = NearestNeighbors(
            n_neighbors=k_neighbors, 
            algorithm='kd_tree',
            metric='euclidean'
        )
        self.knn_model.fit(self.training_features)
        
        return self.deviation_model, well_stats
    
    def _classify_section(self, inc_array):
        """Classifica secao do poco baseado em inclinacao"""
        conditions = [
            inc_array < 3,
            (inc_array >= 3) & (inc_array < 30),
            (inc_array >= 30) & (inc_array < 60),
            inc_array >= 60
        ]
        choices = ["vertical", "buildup", "tangent", "horizontal"]
        return np.select(conditions, choices, default="tangent")
    
    def apply_tortuosity(self, planned_df, max_dls=10.0, smoothing_passes=2):
        """Aplica tortuosidade aprendida em nova trajetoria"""
        if self.knn_model is None:
            raise ValueError("Modelo nao treinado. Execute build_deviation_model primeiro.")
        
        # Carrega e uniformiza trajetoria planejada
        plan = self.load_trajectory_from_df(planned_df)
        
        # Normaliza MD
        md_range = plan["MD"].max() - plan["MD"].min()
        plan["MD_norm"] = (plan["MD"] - plan["MD"].min()) / md_range if md_range > 1e-3 else 0
        
        # === INFERENCIA VETORIZADA COM KNN ===
        query_features = plan[["MD_norm", "Inc"]].values
        query_features_scaled = self.feature_scaler.transform(query_features)
        
        # Busca vizinhos
        distances, indices = self.knn_model.kneighbors(query_features_scaled)
        
        # Ponderacao IDW (Inverse Distance Weighting)
        distances = np.maximum(distances, 1e-8)  # Evita divisao por zero
        weights = 1.0 / distances
        sum_weights = np.sum(weights, axis=1, keepdims=True)
        
        # Media ponderada dos deltas dos vizinhos
        neighbor_delta_inc = self.deviation_model.iloc[indices.flatten()]["delta_inc"].values.reshape(indices.shape)
        neighbor_delta_azi = self.deviation_model.iloc[indices.flatten()]["delta_azi"].values.reshape(indices.shape)
        
        mean_delta_inc = np.sum(neighbor_delta_inc * weights, axis=1) / sum_weights.flatten()
        mean_delta_azi = np.sum(neighbor_delta_azi * weights, axis=1) / sum_weights.flatten()
        
        # Desvio padrao local (para adicao de ruido)
        std_delta_inc = np.std(neighbor_delta_inc, axis=1)
        std_delta_azi = np.std(neighbor_delta_azi, axis=1)
        
        plan["delta_inc"] = mean_delta_inc
        plan["delta_azi"] = mean_delta_azi
        
        # === ADICAO DE RUIDO (se configurado) ===
        if self.mwd_noise_mode == "add":
            sig_inc, sig_azi = self._get_dynamic_uncertainty(plan["Inc"].values)
            
            # Combina variabilidade local + incerteza instrumental
            noise_scale_inc = np.sqrt(std_delta_inc**2 + (sig_inc * self.mwd_noise_factor)**2)
            noise_scale_azi = np.sqrt(std_delta_azi**2 + (sig_azi * self.mwd_noise_factor)**2)
            
            plan["delta_inc"] += np.random.normal(0, noise_scale_inc * 0.3, size=len(plan))
            plan["delta_azi"] += np.random.normal(0, noise_scale_azi * 0.3, size=len(plan))
        
        # Aplica deltas
        plan["Inc_adjusted"] = np.clip(plan["Inc"] + plan["delta_inc"], 0, 120)  # Permite overshoot ate 120
        plan["Azi_adjusted"] = (plan["Azi"] + plan["delta_azi"]) % 360
        
        # === SUAVIZACAO COM PRESERVACAO DE NORMA ===
        for _ in range(smoothing_passes):
            # Inclinacao: filtro gaussiano simples
            plan["Inc_adjusted"] = gaussian_filter1d(plan["Inc_adjusted"], sigma=1.5)
            
            # Azimute: suavizacao vetorial com renormalizacao
            azi_rad = np.radians(plan["Azi_adjusted"])
            sin_az = np.sin(azi_rad)
            cos_az = np.cos(azi_rad)
            
            sin_az_smooth = gaussian_filter1d(sin_az, sigma=1.5)
            cos_az_smooth = gaussian_filter1d(cos_az, sigma=1.5)
            
            # Renormaliza para manter vetor unitario
            norm = np.sqrt(sin_az_smooth**2 + cos_az_smooth**2)
            sin_az_smooth /= norm
            cos_az_smooth /= norm
            
            plan["Azi_adjusted"] = np.degrees(np.arctan2(sin_az_smooth, cos_az_smooth)) % 360
        
        # Clip final de inclinacao
        plan["Inc_adjusted"] = np.clip(plan["Inc_adjusted"], 0, 90)
        
        # === LIMITADOR DE DLS (com recalculo iterativo) ===
        inc_adj = plan["Inc_adjusted"].values.copy()
        azi_adj = plan["Azi_adjusted"].values.copy()
        md = plan["MD"].values
        
        dls_violations = 0
        
        for i in range(1, len(plan)):
            dmd = md[i] - md[i-1]
            if dmd <= 1e-6:
                continue
            
            # Calcula DLS deste intervalo
            i1, i2 = np.radians(inc_adj[i-1]), np.radians(inc_adj[i])
            a1, a2 = np.radians(azi_adj[i-1]), np.radians(azi_adj[i])
            
            cos_dls = np.clip(
                np.cos(i1)*np.cos(i2) + np.sin(i1)*np.sin(i2)*np.cos(a2-a1), 
                -1, 1
            )
            dls_val = np.degrees(np.arccos(cos_dls)) / dmd * 30
            
            # Se excedeu o limite, reduz proporcionalmente
            if dls_val > max_dls:
                ratio = max_dls / dls_val
                
                inc_adj[i] = inc_adj[i-1] + (inc_adj[i] - inc_adj[i-1]) * ratio
                
                # Delta azimute com wrapping
                diff_azi = azi_adj[i] - azi_adj[i-1]
                diff_azi = ((diff_azi + 180) % 360) - 180
                azi_adj[i] = (azi_adj[i-1] + diff_azi * ratio) % 360
                
                dls_violations += 1
        
        plan["Inc_adjusted"] = inc_adj
        plan["Azi_adjusted"] = azi_adj
        
        # === RECALCULO COMPLETO DE COORDENADAS 3D ===
        survey_adjusted = MinimumCurvature.calculate_survey(
            plan["MD"].values,
            plan["Inc_adjusted"].values,
            plan["Azi_adjusted"].values,
            surface_coords=(plan["N"].iloc[0], plan["E"].iloc[0], plan["TVD"].iloc[0])
        )
        
        # Calcula metricas de drift
        drift_tvd = abs(survey_adjusted["TVD"].iloc[-1] - plan["TVD"].iloc[-1])
        drift_north = abs(survey_adjusted["N"].iloc[-1] - plan["N"].iloc[-1])
        drift_east = abs(survey_adjusted["E"].iloc[-1] - plan["E"].iloc[-1])
        drift_horizontal = np.sqrt(drift_north**2 + drift_east**2)
        
        # Adiciona metricas ao resultado
        survey_adjusted.attrs["drift_tvd"] = drift_tvd
        survey_adjusted.attrs["drift_horizontal"] = drift_horizontal
        survey_adjusted.attrs["drift_north"] = drift_north
        survey_adjusted.attrs["drift_east"] = drift_east
        survey_adjusted.attrs["dls_violations"] = dls_violations
        survey_adjusted.attrs["dls_violation_rate"] = dls_violations / len(plan) * 100
        
        return survey_adjusted


def parse_trajectory_file(uploaded_file):
    """Lê arquivos forçando o padrão de colunas conhecido (Seq, MD, Inc, Azi...)"""
    trajectories = []
    
    try:
        # Garante que o nome do arquivo seja string
        file_name = str(uploaded_file.name)
        
        if file_name.endswith(".csv"):
            content = uploaded_file.read().decode("utf-8")
            # Usa o separador e skiprows que funcionavam na sua versão anterior
            df = pd.read_csv(io.StringIO(content), sep=";", decimal=",", skiprows=2)
            
            # Renomeia forçadamente as colunas (lógica da versão anterior)
            col_names = ["Seq", "MD", "Inc", "Azi", "TVD", "COTA", "Vertical", 
                         "Displ_NS", "Displ_EW", "DLS", "UTM_Y", "UTM_X"]
            
            # Garante que não dê erro se o arquivo tiver menos colunas que a lista
            limit = min(len(df.columns), len(col_names))
            df.columns.values[:limit] = col_names[:limit]
            
            # Filtra apenas o necessário, se as colunas existirem
            cols_to_keep = [c for c in ["MD", "Inc", "Azi", "TVD", "DLS"] if c in df.columns]
            if len(cols_to_keep) >= 3: # Mínimo MD, Inc, Azi
                trajectories.append((file_name, df[cols_to_keep]))
        
        elif file_name.endswith((".xlsx", ".xls")):
            excel_file = pd.ExcelFile(uploaded_file)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=2)
                
                if len(df.columns) >= 4:
                    col_names = ["Seq", "MD", "Inc", "Azi", "TVD", "COTA", "Vertical", 
                                 "Displ_NS", "Displ_EW", "DLS", "UTM_Y", "UTM_X"]
                    
                    limit = min(len(df.columns), len(col_names))
                    df.columns.values[:limit] = col_names[:limit]
                    
                    cols_to_keep = ["MD", "Inc", "Azi", "TVD"]
                    if "DLS" in df.columns:
                        cols_to_keep.append("DLS")
                    
                    # CORREÇÃO DO ERRO: str(sheet_name)
                    # Converte o nome da aba para texto para evitar erro de concatenação
                    trajectories.append((str(sheet_name), df[cols_to_keep]))

    except Exception as e:
        st.error(f"Erro ao processar {uploaded_file.name}: {str(e)}")
    
    return trajectories

def _standardize_columns(df):
    """Padroniza nomes de colunas para MD, Inc, Azi, TVD"""
    cols_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if any(x in col_lower for x in ["md", "depth", "measured"]):
            cols_map[col] = "MD"
        elif "inc" in col_lower:
            cols_map[col] = "Inc"
        elif any(x in col_lower for x in ["azi", "azm", "azimuth"]):
            cols_map[col] = "Azi"
        elif "tvd" in col_lower:
            cols_map[col] = "TVD"
        elif "dls" in col_lower:
            cols_map[col] = "DLS"
    
    return df.rename(columns=cols_map)


def _validate_trajectory(df):
    """Valida se DataFrame tem colunas minimas necessarias"""
    required = {"MD", "Inc", "Azi"}
    return required.issubset(df.columns) and len(df) >= 2


def export_to_csv(df):
    """Exporta trajetoria para formato CSV padrao"""
    output = io.StringIO()
    output.write("Seq;Measured;Incl;Azimuth;TVD;Northing;Easting;DLS\n")
    output.write("#;m;deg;deg;m;m;m;deg/30m\n")
    output.write("===;========;======;=======;======;========;========;========\n")
    
    for i, row in df.iterrows():
        line = (f"{i+1};{row['MD']:.2f};{row['Inc']:.2f};{row['Azi']:.2f};"
               f"{row['TVD']:.2f};{row['N']:.2f};{row['E']:.2f};{row['DLS']:.2f}\n")
        output.write(line.replace(".", ","))
    
    return output.getvalue()


def main():
    st.set_page_config(page_title="Gerador de Tortuosidade - Versão Completa", layout="wide")
    
    st.title("🛢️ Gerador de Tortuosidade para Simulação de Desgaste")
    st.caption("Versão Profissional com Minimum Curvature e Análise Completa de Drift")
    
    with st.expander("📘 Metodologia Implementada", expanded=False):
        st.markdown("""
        ### Implementação Completa
        
        **1. Interpolação Correta de Ângulos**
        - Inclinação: Interpolação linear (evita oscilações artificiais)
        - Azimute: SLERP esférica (trata descontinuidade 0°/360° corretamente)
        
        **2. Cálculo de Coordenadas 3D**
        - Método: **Minimum Curvature** (padrão da indústria SPE)
        - Calcula: TVD, N/S, E/W após aplicar tortuosidade
        
        **3. Modelo de Incerteza Dinâmica (ISCWSA)**
        - σ(Inc) = 0.15° × (1 + 0.06 × sin(Inc))
        - σ(Azi) = 0.40° × (1 + 0.06 × sin(Inc))
        - Maior em poços horizontais (onde importa mais)
        
        **4. Busca de Vizinhos (KNN)**
        - Features escalonadas: MD normalizado + Inclinação
        - Ponderação IDW (pontos mais próximos têm mais peso)
        - k adaptativo: 5-20 vizinhos dependendo do dataset
        
        **5. Análise de Drift Completa**
        - Drift Vertical (TVD): Impacto na profundidade do target
        - Drift Horizontal (N/S, E/W): Deslocamento lateral do alvo geológico
        - Alertas automáticos se drift > 5m em qualquer direção
        
        **6. Limitador de DLS**
        - Reduz dog-legs irrealistas mantendo direção geral
        - Recalcula coordenadas 3D após limitação
        - Reporta taxa de violações (% de intervalos corrigidos)
        """)
    
    st.divider()
    
    # === SIDEBAR DE PARAMETROS ===
    st.sidebar.header("⚙️ Parâmetros de Configuração")
    
    md_spacing = st.sidebar.number_input(
        "Espaçamento MD (m)",
        min_value=1.0, max_value=30.0, value=10.0, step=1.0,
        help="Resolução da interpolação. Menor = mais detalhe (mais lento)"
    )
    
    smoothing_window = st.sidebar.slider(
        "Suavização Histórico (σ)",
        min_value=0, max_value=10, value=3,
        help="Remove ruído dos dados históricos antes do aprendizado"
    )
    
    max_dls = st.sidebar.number_input(
        "DLS Máximo (°/30m)",
        min_value=3.0, max_value=20.0, value=10.0, step=0.5,
        help="Limite físico do BHA. Típico: 8-12 para BHA convencional"
    )
    
    smoothing_passes = st.sidebar.slider(
        "Passes de Suavização Final",
        min_value=0, max_value=5, value=2,
        help="Suaviza curva gerada (reduz picos de DLS)"
    )
    
    st.sidebar.divider()
    st.sidebar.subheader("🔬 Tratamento de Incerteza MWD")
    
    mwd_mode = st.sidebar.radio(
        "Modo de Operação",
        options=["subtract", "add"],
        format_func=lambda x: {
            "subtract": "🔵 Subtrair (Otimista)",
            "add": "🔴 Adicionar (Conservador)"
        }[x],
        help="Subtract: Remove ruído MWD dos históricos | Add: Adiciona ruído à predição"
    )
    
    mwd_factor = st.sidebar.slider(
        "Fator de Incerteza (%)",
        min_value=0, max_value=100, value=50,
        help="Porcentagem da incerteza ISCWSA a aplicar"
    ) / 100.0
    
    st.sidebar.divider()
    st.sidebar.info(f"""
    **Configuração Atual:**
    - Resolução: {md_spacing:.1f}m
    - DLS Max: {max_dls:.1f}°/30m
    - Modo MWD: {mwd_mode.upper()}
    - Fator: {mwd_factor*100:.0f}%
    - Suavização: {smoothing_passes} passes
    """)
    
    # === UPLOAD DE ARQUIVOS ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📂 Trajetórias Planejadas (Histórico)")
        f_planned = st.file_uploader(
            "Upload CSV ou Excel",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="planned",
            help="Trajetórias originalmente planejadas dos poços de correlação"
        )
    
    with col2:
        st.subheader("📂 Trajetórias Executadas (Histórico)")
        f_executed = st.file_uploader(
            "Upload CSV ou Excel",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="executed",
            help="Trajetórias reais perfuradas (com tortuosidade)"
        )
    
    st.divider()
    
    st.subheader("🎯 Trajetória Alvo (Novo Poço)")
    f_target = st.file_uploader(
        "Upload da trajetória planejada do poço a ser simulado",
        type=["csv", "xlsx", "xls"],
        key="target"
    )
    
    # === PROCESSAMENTO ===
    if st.button("🚀 Processar Trajetória", type="primary", use_container_width=True):
        if not (f_planned and f_executed and f_target):
            st.error("❌ Carregue todos os arquivos necessários!")
            st.stop()
        
        try:
            with st.spinner("⏳ Lendo arquivos e construindo modelo..."):
                # Parse de arquivos
                planned_trajs = []
                for f in f_planned:
                    trajs = parse_trajectory_file(f)
                    planned_trajs.extend([t[1] for t in trajs])
                
                executed_trajs = []
                for f in f_executed:
                    trajs = parse_trajectory_file(f)
                    executed_trajs.extend([t[1] for t in trajs])
                
                target_trajs = parse_trajectory_file(f_target)
                if not target_trajs:
                    st.error("❌ Arquivo alvo inválido ou sem dados!")
                    st.stop()
                
                target_df = target_trajs[0][1]
                
                # Validacao
                if len(planned_trajs) != len(executed_trajs):
                    st.warning(f"⚠️ Número diferente de trajetórias (Planejadas: {len(planned_trajs)}, Executadas: {len(executed_trajs)})")
                
                # Inicializa modelo
                model = SingleTrajectoryTortuosity(
                    md_spacing=md_spacing,
                    smoothing_window=smoothing_window,
                    mwd_noise_mode=mwd_mode,
                    mwd_noise_factor=mwd_factor
                )
                
                # Treino
                st.info("🔄 Calibrando modelo com poços históricos...")
                deviation_model, well_stats = model.build_deviation_model(
                    planned_trajs, executed_trajs
                )
                
                st.success(f"✅ Modelo treinado com {len(deviation_model)} pontos de {len(well_stats)} poços")
            
            with st.spinner("⏳ Aplicando tortuosidade e recalculando coordenadas 3D..."):
                result = model.apply_tortuosity(
                    target_df, 
                    max_dls=max_dls, 
                    smoothing_passes=smoothing_passes
                )
            
            st.success("✅ Processamento concluído!")
            
            # === DASHBOARD DE RESULTADOS ===
            st.divider()
            st.header("📊 Resultados da Análise")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Análise Principal",
                "📉 Estatísticas do Modelo",
                "⚠️ Análise de Drift",
                "💾 Exportar Dados"
            ])
            
            # === TAB 1: ANALISE PRINCIPAL ===
            with tab1:
                st.subheader("Métricas de Tortuosidade")
                
                col1, col2, col3, col4 = st.columns(4)
                
                dls_original_mean = target_df["DLS"].mean() if "DLS" in target_df.columns else 0
                dls_adjusted_mean = result["DLS"].mean()
                dls_adjusted_max = result["DLS"].max()
                
                increment = ((dls_adjusted_mean / dls_original_mean - 1) * 100) if dls_original_mean > 0 else 0
                
                col1.metric(
                    "DLS Médio Original",
                    f"{dls_original_mean:.2f} °/30m",
                    help="DLS da trajetória planejada"
                )
                
                col2.metric(
                    "DLS Médio Ajustado",
                    f"{dls_adjusted_mean:.2f} °/30m",
                    delta=f"{increment:+.1f}%" if dls_original_mean > 0 else None,
                    help="DLS após aplicar tortuosidade"
                )
                
                col3.metric(
                    "DLS Máximo Gerado",
                    f"{dls_adjusted_max:.2f} °/30m",
                    delta=f"{dls_adjusted_max - max_dls:.2f}" if dls_adjusted_max > max_dls * 0.95 else None,
                    delta_color="inverse",
                    help="Pico de DLS na trajetória ajustada"
                )
                
                violation_rate = result.attrs.get("dls_violation_rate", 0)
                col4.metric(
                    "Taxa de Correções DLS",
                    f"{violation_rate:.1f}%",
                    help="Percentual de intervalos que excederam o limite e foram corrigidos"
                )
                
                # Avisos
                if dls_adjusted_max > max_dls * 0.95:
                    st.warning(f"⚠️ DLS máximo ({dls_adjusted_max:.2f}) muito próximo ao limite ({max_dls:.2f}). Considere aumentar o limite ou aplicar mais suavização.")
                
                if violation_rate > 20:
                    st.warning(f"⚠️ Taxa elevada de correções DLS ({violation_rate:.1f}%). A tortuosidade histórica pode ser incompatível com o limite configurado.")
                
                st.divider()
                
                # Graficos
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.subheader("Perfil de DLS")
                    chart_data = result[["MD", "DLS"]].copy()
                    chart_data["DLS_Limite"] = max_dls
                    st.line_chart(chart_data.set_index("MD"))
                
                with col_b:
                    st.subheader("Distribuição de Inclinação")
                    st.bar_chart(result["Inc"].value_counts(bins=10).sort_index())
                
                st.divider()
                st.subheader("Visualização de Dados (Primeiras 30 Linhas)")
                display_cols = ["MD", "Inc", "Azi", "TVD", "N", "E", "DLS"]
                st.dataframe(
                    result[display_cols].head(30).style.format({
                        "MD": "{:.2f}",
                        "Inc": "{:.2f}",
                        "Azi": "{:.2f}",
                        "TVD": "{:.2f}",
                        "N": "{:.2f}",
                        "E": "{:.2f}",
                        "DLS": "{:.2f}"
                    }),
                    use_container_width=True,
                    height=400
                )
            
            # === TAB 2: ESTATISTICAS DO MODELO ===
            with tab2:
                st.subheader("Estatísticas dos Poços de Correlação")
                
                stats_df = pd.DataFrame(well_stats).T
                stats_display = stats_df[["pontos", "delta_inc_mean", "delta_inc_std", 
                                         "delta_azi_mean", "delta_azi_std"]]
                
                st.dataframe(
                    stats_display.style.format({
                        "pontos": "{:.0f}",
                        "delta_inc_mean": "{:.3f}°",
                        "delta_inc_std": "{:.3f}°",
                        "delta_azi_mean": "{:.3f}°",
                        "delta_azi_std": "{:.3f}°"
                    }),
                    use_container_width=True
                )
                
                st.divider()
                st.subheader("Distribuição de Seções nos Dados Históricos")
                
                section_stats = deviation_model.groupby("section").agg({
                    "delta_inc": ["count", "mean", "std"],
                    "delta_azi": ["mean", "std"]
                }).round(3)
                
                st.dataframe(section_stats, use_container_width=True)
                
                st.divider()
                st.subheader("Amostra do Modelo de Desvios (50 pontos aleatórios)")
                st.dataframe(
                    deviation_model.sample(min(50, len(deviation_model))),
                    use_container_width=True,
                    height=400
                )
            
            # === TAB 3: ANALISE DE DRIFT ===
            with tab3:
                st.subheader("🎯 Análise de Deslocamento do Target")
                
                drift_tvd = result.attrs.get("drift_tvd", 0)
                drift_horizontal = result.attrs.get("drift_horizontal", 0)
                drift_north = result.attrs.get("drift_north", 0)
                drift_east = result.attrs.get("drift_east", 0)
                
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric(
                    "Drift Vertical (TVD)",
                    f"{drift_tvd:.2f} m",
                    delta=f"{drift_tvd:.2f} m" if drift_tvd > 5 else None,
                    delta_color="inverse",
                    help="Diferença de profundidade vertical final"
                )
                
                col2.metric(
                    "Drift Horizontal Total",
                    f"{drift_horizontal:.2f} m",
                    delta=f"{drift_horizontal:.2f} m" if drift_horizontal > 10 else None,
                    delta_color="inverse",
                    help="Deslocamento lateral total (√(N²+E²))"
                )
                
                col3.metric(
                    "Drift Norte/Sul",
                    f"{drift_north:+.2f} m",
                    help="Componente N/S do deslocamento"
                )
                
                col4.metric(
                    "Drift Leste/Oeste",
                    f"{drift_east:+.2f} m",
                    help="Componente E/W do deslocamento"
                )
                
                st.divider()
                
                # Alertas de Drift
                critical_drift = False
                
                if drift_tvd > 10:
                    st.error(f"🔴 **DRIFT CRÍTICO EM TVD:** {drift_tvd:.2f}m de diferença na profundidade final! Verifique se o alvo geológico ainda está acessível.")
                    critical_drift = True
                elif drift_tvd > 5:
                    st.warning(f"🟡 **Drift Moderado em TVD:** {drift_tvd:.2f}m. Recomenda-se revisão do target.")
                else:
                    st.success(f"🟢 Drift em TVD aceitável: {drift_tvd:.2f}m")
                
                if drift_horizontal > 20:
                    st.error(f"🔴 **DRIFT CRÍTICO HORIZONTAL:** {drift_horizontal:.2f}m de deslocamento lateral! O poço pode estar fora da zona produtora.")
                    critical_drift = True
                elif drift_horizontal > 10:
                    st.warning(f"🟡 **Drift Horizontal Moderado:** {drift_horizontal:.2f}m. Verifique limites da zona.")
                else:
                    st.success(f"🟢 Drift horizontal aceitável: {drift_horizontal:.2f}m")
                
                if critical_drift:
                    st.error("""
                    ### ⚠️ AÇÃO RECOMENDADA
                    O drift calculado indica que a tortuosidade pode ter deslocado significativamente o poço do target planejado.
                    
                    **Opções:**
                    1. Reduzir o fator de ruído MWD
                    2. Aumentar os passes de suavização
                    3. Revisar os poços de correlação (podem ter tortuosidade atípica)
                    4. Aceitar o drift se for realista para a área
                    """)
                
                st.divider()
                st.subheader("Comparação de Coordenadas Finais")
                
                comparison = pd.DataFrame({
                    "Métrica": ["TVD Final", "Northing Final", "Easting Final"],
                    "Planejado": [
                        target_df["TVD"].iloc[-1] if "TVD" in target_df.columns else result["TVD"].iloc[0],
                        0,  # Assumindo origem em (0,0)
                        0
                    ],
                    "Ajustado": [
                        result["TVD"].iloc[-1],
                        result["N"].iloc[-1],
                        result["E"].iloc[-1]
                    ]
                })
                
                comparison["Diferença"] = comparison["Ajustado"] - comparison["Planejado"]
                
                st.dataframe(
                    comparison.style.format({
                        "Planejado": "{:.2f} m",
                        "Ajustado": "{:.2f} m",
                        "Diferença": "{:+.2f} m"
                    }),
                    use_container_width=True
                )
            
            # === TAB 4: EXPORTAR ===
            with tab4:
                st.subheader("💾 Download dos Resultados")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📄 Formato CSV (Padrão)")
                    csv_output = export_to_csv(result)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_output,
                        file_name="trajetoria_com_tortuosidade.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.caption("Formato compatível com softwares de perfuração")
                
                with col2:
                    st.markdown("#### 📊 Formato Excel (Completo)")
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        result.to_excel(writer, sheet_name="Trajetoria_Ajustada", index=False)
                        deviation_model.to_excel(writer, sheet_name="Modelo_Desvios", index=False)
                        
                        # Adiciona sheet de resumo
                        summary = pd.DataFrame({
                            "Parametro": [
                                "MD Spacing (m)",
                                "DLS Maximo (deg/30m)",
                                "Modo MWD",
                                "Fator MWD (%)",
                                "Passes Suavizacao",
                                "Drift TVD (m)",
                                "Drift Horizontal (m)",
                                "DLS Medio Original",
                                "DLS Medio Ajustado",
                                "DLS Maximo Gerado",
                                "Taxa Violacoes DLS (%)"
                            ],
                            "Valor": [
                                md_spacing,
                                max_dls,
                                mwd_mode,
                                mwd_factor * 100,
                                smoothing_passes,
                                drift_tvd,
                                drift_horizontal,
                                dls_original_mean,
                                dls_adjusted_mean,
                                dls_adjusted_max,
                                violation_rate
                            ]
                        })
                        summary.to_excel(writer, sheet_name="Resumo", index=False)
                    
                    st.download_button(
                        label="⬇️ Download Excel Completo",
                        data=buffer.getvalue(),
                        file_name="analise_completa_tortuosidade.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    st.caption("Inclui trajetória, modelo de desvios e resumo executivo")
                
                st.divider()
                st.info("""
                ### 📋 Conteúdo dos Arquivos
                
                **CSV:**
                - Trajetória ajustada com coordenadas 3D completas
                - Colunas: Seq, MD, Inc, Azi, TVD, N, E, DLS
                
                **Excel:**
                - **Sheet 1:** Trajetória ajustada (mesma do CSV)
                - **Sheet 2:** Modelo de desvios históricos usado no treinamento
                - **Sheet 3:** Resumo executivo com todos os parâmetros e métricas
                """)
        
        except Exception as e:
            st.error(f"❌ Erro durante o processamento: {str(e)}")
            with st.expander("🔍 Detalhes do Erro (Para Debug)"):
                st.exception(e)


if __name__ == "__main__":
    main()