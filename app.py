import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import io
import zipfile
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


class SingleTrajectoryTortuosity:
    def __init__(self, md_spacing, smoothing_window, mwd_noise_mode, mwd_noise_factor):
        self.md_spacing = md_spacing
        self.smoothing_window = smoothing_window
        self.mwd_noise_mode = mwd_noise_mode
        self.mwd_noise_factor = mwd_noise_factor
        self.sigma_inc_mwd = 0.15
        self.sigma_azi_mwd = 0.40
        self.deviation_model = None
    
    
    def load_trajectory_from_df(self, df):
        for col in ["MD", "Inc", "Azi", "TVD"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna(subset=["MD", "Inc", "Azi"]).reset_index(drop=True)
        
        md_uniform = np.arange(df["MD"].min(), df["MD"].max() + self.md_spacing, self.md_spacing)
        
        inc_interp = interp1d(df["MD"], df["Inc"], kind="cubic", fill_value="extrapolate")
        azi_interp = interp1d(df["MD"], df["Azi"], kind="cubic", fill_value="extrapolate")
        tvd_interp = interp1d(df["MD"], df["TVD"], kind="cubic", fill_value="extrapolate")
        
        df_uniform = pd.DataFrame({
            "MD": md_uniform,
            "Inc": inc_interp(md_uniform),
            "Azi": azi_interp(md_uniform),
            "TVD": tvd_interp(md_uniform)
        })
        
        df_uniform["Azi"] = df_uniform["Azi"] % 360
        return df_uniform
    
    
    @staticmethod
    def calc_dls(p1, p2):
        i1, i2 = np.radians(p1["Inc"]), np.radians(p2["Inc"])
        a1, a2 = np.radians(p1["Azi"]), np.radians(p2["Azi"])
        cos_dls = np.cos(i1)*np.cos(i2) + np.sin(i1)*np.sin(i2)*np.cos(a2-a1)
        cos_dls = np.clip(cos_dls, -1, 1)
        dls_rad = np.arccos(cos_dls)
        dmd = p2["MD"] - p1["MD"]
        return np.degrees(dls_rad) / dmd * 30 if dmd > 0 else 0
    
    
    def build_deviation_model(self, planned_dfs, executed_dfs):
        all_deviations = []
        well_stats = {}
        
        for idx, (plan, exec_) in enumerate(zip(planned_dfs, executed_dfs)):
            plan_uniform = self.load_trajectory_from_df(plan)
            exec_uniform = self.load_trajectory_from_df(exec_)
            
            merged = pd.merge_asof(plan_uniform, exec_uniform, on="MD", 
                                   suffixes=("_plan", "_exec"),
                                   direction="nearest", tolerance=self.md_spacing)
            merged = merged.dropna()
            
            merged["delta_inc"] = merged["Inc_exec"] - merged["Inc_plan"]
            merged["delta_azi"] = merged["Azi_exec"] - merged["Azi_plan"]
            
            merged["delta_azi"] = merged["delta_azi"].apply(
                lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x)
            )
            
            if self.mwd_noise_mode == "subtract":
                merged["delta_inc"] = merged["delta_inc"].apply(
                    lambda x: np.sign(x) * max(0, abs(x) - self.sigma_inc_mwd * self.mwd_noise_factor)
                )
                merged["delta_azi"] = merged["delta_azi"].apply(
                    lambda x: np.sign(x) * max(0, abs(x) - self.sigma_azi_mwd * self.mwd_noise_factor)
                )
            
            if self.smoothing_window > 0:
                merged["delta_inc"] = gaussian_filter1d(merged["delta_inc"], sigma=self.smoothing_window)
                merged["delta_azi"] = gaussian_filter1d(merged["delta_azi"], sigma=self.smoothing_window)
            
            md_range = merged["MD"].max() - merged["MD"].min()
            merged["MD_norm"] = (merged["MD"] - merged["MD"].min()) / md_range
            merged["section"] = merged.apply(self._classify_section, axis=1)
            
            deviations = merged[["MD_norm", "Inc_plan", "Azi_plan", 
                                "delta_inc", "delta_azi", "section"]].copy()
            deviations.columns = ["MD_norm", "Inc", "Azi", "delta_inc", "delta_azi", "section"]
            
            all_deviations.append(deviations)
            
            well_stats[f"Poco {idx+1}"] = {
                "pontos": len(deviations),
                "delta_inc_mean": deviations["delta_inc"].mean(),
                "delta_inc_std": deviations["delta_inc"].std(),
                "delta_azi_mean": deviations["delta_azi"].mean(),
                "delta_azi_std": deviations["delta_azi"].std()
            }
        
        self.deviation_model = pd.concat(all_deviations, ignore_index=True)
        return self.deviation_model, well_stats
    
    
    def _classify_section(self, row):
        inc = row["Inc_plan"]
        if inc < 3:
            return "vertical"
        elif 3 <= inc < 30:
            return "buildup"
        elif 30 <= inc < 60:
            return "tangent"
        else:
            return "horizontal"
    
    
    def apply_tortuosity(self, planned_df):
        plan = self.load_trajectory_from_df(planned_df)
        
        md_range = plan["MD"].max() - plan["MD"].min()
        plan["MD_norm"] = (plan["MD"] - plan["MD"].min()) / md_range
        plan["section"] = plan.apply(lambda row: self._classify_section_simple(row["Inc"]), axis=1)
        
        plan["delta_inc"] = 0.0
        plan["delta_azi"] = 0.0
        
        for i, row in plan.iterrows():
            similar = self._find_similar_points(row)
            
            if len(similar) > 0:
                plan.loc[i, "delta_inc"] = similar["delta_inc"].mean()
                plan.loc[i, "delta_azi"] = similar["delta_azi"].mean()
                
                if self.mwd_noise_mode == "add":
                    noise_inc = (similar["delta_inc"].std() + 
                                self.sigma_inc_mwd * self.mwd_noise_factor)
                    noise_azi = (similar["delta_azi"].std() + 
                                self.sigma_azi_mwd * self.mwd_noise_factor)
                    
                    plan.loc[i, "delta_inc"] += np.random.normal(0, noise_inc * 0.3)
                    plan.loc[i, "delta_azi"] += np.random.normal(0, noise_azi * 0.3)
        
        plan["Inc_adjusted"] = np.clip(plan["Inc"] + plan["delta_inc"], 0, 90)
        plan["Azi_adjusted"] = (plan["Azi"] + plan["delta_azi"]) % 360
        
        plan["DLS"] = 0.0
        for i in range(1, len(plan)):
            p1 = pd.Series({"Inc": plan.loc[i-1, "Inc_adjusted"], 
                           "Azi": plan.loc[i-1, "Azi_adjusted"],
                           "MD": plan.loc[i-1, "MD"]})
            p2 = pd.Series({"Inc": plan.loc[i, "Inc_adjusted"], 
                           "Azi": plan.loc[i, "Azi_adjusted"],
                           "MD": plan.loc[i, "MD"]})
            plan.loc[i, "DLS"] = self.calc_dls(p1, p2)
        
        return plan
    
    
    def _find_similar_points(self, point):
        mask = (
            (np.abs(self.deviation_model["MD_norm"] - point["MD_norm"]) < 0.1) &
            (np.abs(self.deviation_model["Inc"] - point["Inc"]) < 5.0) &
            (self.deviation_model["section"] == point["section"])
        )
        
        if mask.sum() < 3:
            mask = (
                (np.abs(self.deviation_model["MD_norm"] - point["MD_norm"]) < 0.2) &
                (self.deviation_model["section"] == point["section"])
            )
        
        return self.deviation_model[mask]
    
    
    def _classify_section_simple(self, inc):
        if inc < 3:
            return "vertical"
        elif inc < 30:
            return "buildup"
        elif inc < 60:
            return "tangent"
        else:
            return "horizontal"


def parse_trajectory_file(uploaded_file):
    trajectories = []
    
    if uploaded_file.name.endswith(".csv"):
        content = uploaded_file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content), sep=";", decimal=",", skiprows=2)
        df.columns = ["Seq", "MD", "Inc", "Azi", "TVD", "COTA", "Vertical", 
                      "Displ_NS", "Displ_EW", "DLS", "UTM_Y", "UTM_X"]
        trajectories.append((uploaded_file.name, df[["MD", "Inc", "Azi", "TVD"]]))
    
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        excel_file = pd.ExcelFile(uploaded_file)
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=2)
            
            if len(df.columns) >= 4:
                df.columns = ["Seq", "MD", "Inc", "Azi", "TVD", "COTA", "Vertical", 
                              "Displ_NS", "Displ_EW", "DLS", "UTM_Y", "UTM_X"][:len(df.columns)]
                trajectories.append((sheet_name, df[["MD", "Inc", "Azi", "TVD"]]))
    
    return trajectories


def export_to_csv(df):
    output = io.StringIO()
    output.write("Seq;Measured;Incl;Azimuth;TVD;DLS\n")
    output.write("#;depth;angle;angle;depth;(deg/30m)\n")
    output.write("===;========;======;=======;======;========\n")
    
    for i, row in df.iterrows():
        line = f"{i+1};{row['MD']:.2f};{row['Inc_adjusted']:.2f};{row['Azi_adjusted']:.2f};{row['TVD']:.2f};{row['DLS']:.2f}\n"
        output.write(line.replace(".", ","))
    
    return output.getvalue()


def main():
    st.set_page_config(page_title="Gerador de Tortuosidade", layout="wide")
    
    st.title("Gerador de Tortuosidade para Simulacao de Desgaste")
    
    with st.expander("Como funciona esta ferramenta?", expanded=False):
        st.markdown("""
        ### Explicacao para Nao-Especialistas
        
        **Problema:** Quando perfuramos um poco de petroleo, a trajetoria real (executada) nunca e igual a planejada. 
        Isso acontece por varios motivos:
        - Formacoes geologicas diferentes empurram a broca em direcoes inesperadas
        - Dificuldades operacionais durante a perfuracao
        - Incertezas nas medicoes das ferramentas (MWD/LWD)
        
        Essas diferencas criam uma tortuosidade - pequenas curvas e desvios ao longo do poco. 
        Isso e importante porque aumenta o desgaste do revestimento por atrito com a coluna de perfuracao.
        
        ---
        
        ### O que esta ferramenta faz?
        
        **Passo 1: Aprendizado**
        - Voce fornece trajetorias planejadas e executadas de pocos ja perfurados (pocos de correlacao)
        - A ferramenta compara essas trajetorias e aprende o padrao de desvios tipicos
        - Exemplo: em secoes verticais, o desvio medio e 0.2 grau em inclinacao
        
        **Passo 2: Aplicacao**
        - Voce fornece a trajetoria planejada de um novo poco
        - A ferramenta aplica os padroes aprendidos nessa nova trajetoria
        - Resultado: uma trajetoria mais realista para simular desgaste
        
        ---
        
        ### Como funciona matematicamente?
        
        1. **Extracao de Desvios:** Para cada ponto da trajetoria executada, calculamos:
           - Delta Inclinacao = Inc_executada - Inc_planejada
           - Delta Azimute = Azi_executada - Azi_planejada
        
        2. **Classificacao por Secao:** Dividimos o poco em:
           - Vertical (Inc < 3 graus)
           - Build-up (3 < Inc < 30 graus)
           - Tangente (30 < Inc < 60 graus)
           - Horizontal (Inc > 60 graus)
        
        3. **Modelagem Estatistica:** Para cada secao, calculamos:
           - Desvio medio (bias sistematico, ex: bit walk)
           - Desvio padrao (variabilidade operacional)
        
        4. **Aplicacao:** Para o novo poco, em cada ponto:
           - Busca pontos similares nos pocos historicos
           - Aplica o desvio medio desses pontos
           - Adiciona ruido proporcional a incerteza MWD
        
        ---
        
        ### Tratamento da Incerteza MWD
        
        As ferramentas de medicao (MWD/LWD) tem incerteza tipica:
        - Inclinacao: 0.15 graus
        - Azimute: 0.40 graus
        
        Voce pode escolher como tratar isso:
        
        **Opcao 1: Adicionar como ruido**
        - Considera a incerteza como parte da tortuosidade real
        - Resultado: trajetoria mais tortuosa (conservador para desgaste)
        
        **Opcao 2: Subtrair (conservador)**
        - Remove a incerteza dos desvios historicos
        - Considera apenas desvios reais acima do erro de medicao
        - Resultado: trajetoria menos tortuosa (otimista para desgaste)
        
        ---
        
        ### Parametros Ajustaveis
        
        - Espacamento (MD): Distancia entre pontos interpolados (10m recomendado)
        - Suavizacao: Remove ruido de alta frequencia dos surveys (janela de 3 pontos tipica)
        - Fator de ruido MWD: Quanto da incerteza aplicar (0-100%)
        """)
    
    st.divider()
    
    st.sidebar.header("Parametros")
    
    md_spacing = st.sidebar.number_input(
        "Espacamento MD (metros)",
        min_value=1.0, max_value=30.0, value=10.0, step=1.0,
        help="Distancia entre pontos interpolados. Menor = mais detalhe."
    )
    
    smoothing_window = st.sidebar.slider(
        "Janela de Suavizacao",
        min_value=0, max_value=10, value=3,
        help="Remove ruido de medicao. 0 = sem suavizacao, 5 = forte."
    )
    
    mwd_noise_mode = st.sidebar.radio(
        "Tratamento de Incerteza MWD",
        options=["add", "subtract"],
        format_func=lambda x: "Adicionar como ruido" if x == "add" else "Subtrair (conservador)",
        help="Como incorporar a incerteza das medicoes MWD"
    )
    
    mwd_noise_factor = st.sidebar.slider(
        "Fator de Ruido MWD (%)",
        min_value=0, max_value=100, value=50,
        help="Porcentagem da incerteza MWD a aplicar"
    ) / 100.0
    
    st.sidebar.divider()
    st.sidebar.info(f"""
    Incerteza MWD Padrao:
    - sigma(Inc) = 0.15 graus
    - sigma(Azi) = 0.40 graus
    
    Modo atual: {mwd_noise_mode.upper()}
    Fator: {mwd_noise_factor*100:.0f}%
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trajetorias Planejadas")
        planned_files = st.file_uploader(
            "CSV ou Excel (multiplas abas)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="planned",
            help="Arquivos no formato: Seq;MD;Inc;Azi;TVD;..."
        )
    
    with col2:
        st.subheader("Trajetorias Executadas")
        executed_files = st.file_uploader(
            "CSV ou Excel (multiplas abas)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="executed",
            help="Mesma ordem dos planejados"
        )
    
    st.divider()
    
    st.subheader("Trajetoria Alvo (Planejada)")
    target_file = st.file_uploader(
        "Upload da trajetoria planejada do novo poco",
        type=["csv", "xlsx", "xls"],
        key="target"
    )
    
    if st.button("Gerar Trajetoria com Tortuosidade", type="primary"):
        if not planned_files or not executed_files:
            st.error("Carregue trajetorias planejadas e executadas!")
            return
        
        if len(planned_files) != len(executed_files):
            st.error("Numero de arquivos planejados e executados deve ser igual!")
            return
        
        if not target_file:
            st.error("Carregue a trajetoria alvo!")
            return
        
        try:
            with st.spinner("Processando..."):
                planned_trajs = []
                for f in planned_files:
                    trajs = parse_trajectory_file(f)
                    planned_trajs.extend([t[1] for t in trajs])
                
                executed_trajs = []
                for f in executed_files:
                    trajs = parse_trajectory_file(f)
                    executed_trajs.extend([t[1] for t in trajs])
                
                target_trajs = parse_trajectory_file(target_file)
                target_df = target_trajs[0][1]
                
                model = SingleTrajectoryTortuosity(
                    md_spacing=md_spacing,
                    smoothing_window=smoothing_window,
                    mwd_noise_mode=mwd_noise_mode,
                    mwd_noise_factor=mwd_noise_factor
                )
                
                st.info("Calibrando modelo com pocos de correlacao...")
                deviation_model, well_stats = model.build_deviation_model(
                    planned_trajs, executed_trajs
                )
                
                st.info("Aplicando tortuosidade na trajetoria alvo...")
                result = model.apply_tortuosity(target_df)
            
            st.success("Processamento concluido!")
            
            tab1, tab2, tab3 = st.tabs(["Resultados", "Estatisticas", "Download"])
            
            with tab1:
                st.subheader("Comparacao: Planejado vs Ajustado")
                
                col1, col2, col3 = st.columns(3)
                
                dls_plan = result["DLS"].iloc[1:].mean()
                dls_max = result["DLS"].max()
                increment = ((dls_plan / target_df["DLS"].iloc[1:].mean() - 1) * 100) if target_df["DLS"].iloc[1:].mean() > 0 else 0
                
                col1.metric("DLS Medio Planejado", f"{target_df['DLS'].iloc[1:].mean():.2f} graus/30m")
                col2.metric("DLS Medio Ajustado", f"{dls_plan:.2f} graus/30m", f"{increment:+.1f}%")
                col3.metric("DLS Maximo Gerado", f"{dls_max:.2f} graus/30m")
                
                st.dataframe(
                    result[["MD", "Inc", "Inc_adjusted", "Azi", "Azi_adjusted", "DLS"]].head(20),
                    use_container_width=True
                )
            
            with tab2:
                st.subheader("Estatisticas dos Pocos de Correlacao")
                
                stats_df = pd.DataFrame(well_stats).T
                st.dataframe(stats_df.style.format({
                    "pontos": "{:.0f}",
                    "delta_inc_mean": "{:.3f} graus",
                    "delta_inc_std": "{:.3f} graus",
                    "delta_azi_mean": "{:.3f} graus",
                    "delta_azi_std": "{:.3f} graus"
                }), use_container_width=True)
                
                st.subheader("Desvios por Secao do Poco")
                section_stats = deviation_model.groupby("section").agg({
                    "delta_inc": ["count", "mean", "std"],
                    "delta_azi": ["mean", "std"]
                }).round(3)
                st.dataframe(section_stats, use_container_width=True)
            
            with tab3:
                csv_output = export_to_csv(result)
                
                st.download_button(
                    label="Download CSV (formato original)",
                    data=csv_output,
                    file_name="trajetoria_com_tortuosidade.csv",
                    mime="text/csv"
                )
                
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    result.to_excel(writer, sheet_name="Trajetoria Ajustada", index=False)
                    deviation_model.to_excel(writer, sheet_name="Modelo de Desvios", index=False)
                
                st.download_button(
                    label="Download Excel (completo)",
                    data=excel_buffer.getvalue(),
                    file_name="resultado_completo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        except Exception as e:
            st.error(f"Erro no processamento: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
