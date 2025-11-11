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

warnings.filterwarnings('ignore')

# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class SingleTrajectoryTortuosity:
    """Gera trajetória com tortuosidade modelada."""

    def __init__(self, md_spacing: float, smoothing_window: int, 
                 mwd_noise_mode: str, mwd_noise_factor: float):
        self.md_spacing = md_spacing
        self.smoothing_window = smoothing_window
        self.mwd_noise_mode = mwd_noise_mode  # 'add' ou 'subtract'
        self.mwd_noise_factor = mwd_noise_factor
        self.sigma_inc_mwd = 0.15
        self.sigma_azi_mwd = 0.40
        self.deviation_model = None

    def load_trajectory_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Carrega trajetória de DataFrame já parseado."""
        for col in ['MD', 'Inc', 'Azi', 'TVD']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['MD', 'Inc', 'Azi']).reset_index(drop=True)
        md_uniform = np.arange(df['MD'].min(), df['MD'].max() + self.md_spacing, self.md_spacing)
        
        inc_interp = interp1d(df['MD'], df['Inc'], kind='cubic', fill_value='extrapolate')
        azi_interp = interp1d(df['MD'], df['Azi'], kind='cubic', fill_value='extrapolate')
        tvd_interp = interp1d(df['MD'], df['TVD'], kind='cubic', fill_value='extrapolate')
        
        df_uniform = pd.DataFrame({
            'MD': md_uniform,
            'Inc': inc_interp(md_uniform),
            'Azi': azi_interp(md_uniform),
            'TVD': tvd_interp(md_uniform)
        })
        df_uniform['Azi'] = df_uniform['Azi'] % 360
        return df_uniform

    @staticmethod
    def calc_dls(p1: pd.Series, p2: pd.Series) -> float:
        i1, i2 = np.radians(p1['Inc']), np.radians(p2['Inc'])
        a1, a2 = np.radians(p1['Azi']), np.radians(p2['Azi'])
        cos_dls = np.cos(i1)*np.cos(i2) + np.sin(i1)*np.sin(i2)*np.cos(a2-a1)
        cos_dls = np.clip(cos_dls, -1, 1)
        dls_rad = np.arccos(cos_dls)
        dmd = p2['MD'] - p1['MD']
        return np.degrees(dls_rad) / dmd * 30 if dmd > 0 else 0

    def build_deviation_model(self, planned_dfs: List[pd.DataFrame], executed_dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
        all_deviations = []
        well_stats = {}
        for idx, (plan, exec_) in enumerate(zip(planned_dfs, executed_dfs)):
            plan_uniform = self.load_trajectory_from_df(plan)
            exec_uniform = self.load_trajectory_from_df(exec_)
            merged = pd.merge_asof(plan_uniform, exec_uniform, on='MD', suffixes=('_plan', '_exec'),
                                   direction='nearest', tolerance=self.md_spacing)
            merged = merged.dropna()
            merged['delta_inc'] = merged['Inc_exec'] - merged['Inc_plan']
            merged['delta_azi'] = merged['Azi_exec'] - merged['Azi_plan']
            merged['delta_azi'] = merged['delta_azi'].apply(
                lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x)
            )
            if self.mwd_noise_mode == 'subtract':
                merged['delta_inc'] = merged['delta_inc'].apply(
                    lambda x: np.sign(x) * max(0, abs(x) - self.sigma_inc_mwd * self.mwd_noise_factor)
                )
                merged['delta_azi'] = merged['delta_azi'].apply(
                    lambda x: np.sign(x) * max(0, abs(x) - self.sigma_azi_mwd * self.mwd_noise_factor)
                )
            if self.smoothing_window > 0:
                merged['delta_inc'] = gaussian_filter1d(merged['delta_inc'], sigma=self.smoothing_window)
                merged['delta_azi'] = gaussian_filter1d(merged['delta_azi'], sigma=self.smoothing_window)
            md_range = merged['MD'].max() - merged['MD'].min()
            merged['MD_norm'] = (merged['MD'] - merged['MD'].min()) / md_range
            merged['section'] = merged.apply(self._classify_section, axis=1)
            deviations = merged[['MD_norm', 'Inc_plan', 'Azi_plan', 'delta_inc', 'delta_azi', 'section']].copy()
            deviations.columns = ['MD_norm', 'Inc', 'Azi', 'delta_inc', 'delta_azi', 'section']
            all_deviations.append(deviations)
            well_stats[f'Poço {idx+1}'] = {
                'pontos': len(deviations),
                'delta_inc_mean': deviations['delta_inc'].mean(),
                'delta_inc_std': deviations['delta_inc'].std(),
                'delta_azi_mean': deviations['delta_azi'].mean(),
                'delta_azi_std': deviations['delta_azi'].std()
            }
        self.deviation_model = pd.concat(all_deviations, ignore_index=True)
        return self.deviation_model, well_stats

    def _classify_section(self, row) -> str:
        inc = row['Inc_plan']
        if inc < 3:
            return 'vertical'
        elif 3 <= inc < 30:
            return 'buildup'
        elif 30 <= inc < 60:
            return 'tangent'
        else:
            return 'horizontal'

    def apply_tortuosity(self, planned_df: pd.DataFrame) -> pd.DataFrame:
        plan = self.load_trajectory_from_df(planned_df)
        md_range = plan['MD'].max() - plan['MD'].min()
        plan['MD_norm'] = (plan['MD'] - plan['MD'].min()) / md_range
        plan['section'] = plan.apply(lambda row: self._classify_section_simple(row['Inc']), axis=1)
        plan['delta_inc'] = 0.0
        plan['delta_azi'] = 0.0
        for i, row in plan.iterrows():
            similar = self._find_similar_points(row)
            if len(similar) > 0:
                plan.loc[i, 'delta_inc'] = similar['delta_inc'].mean()
                plan.loc[i, 'delta_azi'] = similar['delta_azi'].mean()
                if self.mwd_noise_mode == 'add':
                    noise_inc = (similar['delta_inc'].std() + self.sigma_inc_mwd * self.mwd_noise_factor)
                    noise_azi = (similar['delta_azi'].std() + self.sigma_azi_mwd * self.mwd_noise_factor)
                    plan.loc[i, 'delta_inc'] += np.random.normal(0, noise_inc * 0.3)
                    plan.loc[i, 'delta_azi'] += np.random.normal(0, noise_azi * 0.3)
        plan['Inc_adjusted'] = np.clip(plan['Inc'] + plan['delta_inc'], 0, 90)
        plan['Azi_adjusted'] = (plan['Azi'] + plan['delta_azi']) % 360
        plan['DLS'] = 0.0
        for i in range(1, len(plan)):
            p1 = pd.Series({'Inc': plan.loc[i-1, 'Inc_adjusted'], 'Azi': plan.loc[i-1, 'Azi_adjusted'], 'MD': plan.loc[i-1, 'MD']})
            p2 = pd.Series({'Inc': plan.loc[i, 'Inc_adjusted'], 'Azi': plan.loc[i, 'Azi_adjusted'], 'MD': plan.loc[i, 'MD']})
            plan.loc[i, 'DLS'] = self.calc_dls(p1, p2)
        return plan

    def _find_similar_points(self, point: pd.Series) -> pd.DataFrame:
        mask = (
            (np.abs(self.deviation_model['MD_norm'] - point['MD_norm']) < 0.1) &
            (np.abs(self.deviation_model['Inc'] - point['Inc']) < 5.0) &
            (self.deviation_model['section'] == point['section'])
        )
        if mask.sum() < 3:
            mask = (
                (np.abs(self.deviation_model['MD_norm'] - point['MD_norm']) < 0.2) &
                (self.deviation_model['section'] == point['section'])
            )
        return self.deviation_model[mask]

    def _classify_section_simple(self, inc: float) -> str:
        if inc < 3:
            return 'vertical'
        elif inc < 30:
            return 'buildup'
        elif inc < 60:
            return 'tangent'
        else:
            return 'horizontal'

if __name__ == '__main__':
    print("Script carregado com sucesso!")
