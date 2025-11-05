# -*- coding: utf-8 -*-
"""
Backend Combinado de IA de Ruleta v3.8
=======================================
Fusiona 'main_neighbors.py' y 'main_apuestas_externas.py' en un único
servidor con modos intercambiables y estadísticas separadas.

MEJORAS v3.8 (Optimización de Acertividad):
- (MEJORA 2a) DIVERSITY_WEIGHT cambiado de 0.4 a 4.0 (ahora tiene peso real)
- (MEJORA 2b) Eliminado el '0' como "seguro" en apuestas outside (matemáticamente incorrecto)
- (MEJORA 2c) LONG_TERM_BALANCE_WEIGHT puesto en 0.0 (estrategia simplificada a Momentum)

MEJORAS v3.7 (Bugfix Crítico de Sintaxis):
- (global) CORREGIDO: Se elimina la línea 'app.post("/delete_last_number",-1)(delete_last_number_v2)'
  que causaba un 'TypeError' al iniciar.
- (delete_last_number) MEJORA: Se implementa la lógica de recálculo correcta
  para 'number_last_seen' directamente dentro de la función, eliminando
  duplicados.

MEJORAS v3.6 (Bugfix Crítico de Datos):
- (process_number) CORREGIDO: Registra números repetidos (ej. 20, 20).

MEJORAS v3.5 (Corrección Lógica '0'):
- (process_number) CORREGIDO: Lógica de victoria universal
  'is_win = number in active_bet["covered_numbers"]'
"""
import uvicorn
import asyncio
import math
import random
import os
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, Counter, defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ==============================================================================
# CONFIGURACIÓN GLOBAL Y CONSTANTES
# ==============================================================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INICIO: ENDPOINT PARA SERVIR HTML (Corrección 404) ---

@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Sirve el archivo HTML principal."""
    html_file_path = "roulette_ai_combined.html"
    
    if os.path.exists(html_file_path):
        with open(html_file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    
    return HTMLResponse(
        content=f"<h1>Error 404: Archivo HTML no encontrado.</h1>"
                f"<p>Asegúrate de que el archivo <strong>'{html_file_path}'</strong> "
                "esté en la misma carpeta que tu script de Python.</p>",
        status_code=404
    )

# --- FIN: ENDPOINT PARA SERVIR HTML ---


ROULETTE_ORDER: List[int] = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8,
    23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12,
    35, 3, 26
]

OUTSIDE_BETS = {
    "RED": [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36],
    "BLACK": [2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35],
    "EVEN": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],
    "ODD": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35],
    "LOW": list(range(1, 19)),  # 1-18
    "HIGH": list(range(19, 37))  # 19-36
}

# --- Configuración del Modo NEIGHBORS ---
IA_CONFIG_NEIGHBORS: Dict[str, Any] = {
    # CONTROL DE FASES (MEJORA v3.0)
    "LEARNING_PHASE_MIN":195,  # Empezar a apostar
    "READY_PHASE_MIN": 190,     # Fase de análisis (50-150)
    "OPTIMAL_DATA_THRESHOLD": 190, # Fase óptima (150+)
    
    "MIN_HISTORY_FOR_ANALYSIS": 50, # Mínimo para find_best_number_bet
    "LOOKBACK": 190,
    "MAX_GALE": 2,
    "COOLDOWN_ROUNDS": 5, # Unificado
    "FORCE_ENTRY": True,
    "NEIGHBOURS": 4, # 4 vecinos -> 9 números por set (1+4+4). 2 sets sin solapamiento = 18.

    # FILTRO DE ESTABILIDAD
    "USE_STABILITY_FILTER": True,
    "STABILITY_WINDOW": 30,
    "VOLATILITY_THRESHOLD": 9.5,

    # PESOS
    "COVERAGE_WEIGHT": 2.0, 
    "COVERAGE_QUALITY_WEIGHT": 10.0, 
    "FREQUENCY_WEIGHT": 8.0,
    "RECENCY_WEIGHT": 7.0, 
    "HOT_ZONE_WEIGHT": 9.0, 
    "DIVERSITY_WEIGHT": 4.0,  # ✅ MEJORA 2a: Cambiado de 0.4 a 4.0
    "COLD_ZONE_BONUS": 0.1, 
    "PATTERN_WEIGHT": 5.0, 
    "CLUSTER_WEIGHT": 6.0,

    # PENALIZACIONES
    "REPEAT_BUFFER_PAIRS": 12, 
    "MAX_CONSEC_SAME_PAIR": 0, 
    "REPEAT_PENALTY": 25.0,
    "OVERLAP_PENALTY_MULTIPLIER": 5.0, 
    "COLD_ZONE_PENALTY": 12.0,

    "ACCURACY_WEIGHT": 30.0, 
    "DECAY_GAMMA": 0.78, 
    "HOT_ZONE_WINDOW": 25,
    "HOT_ZONE_THRESHOLD": 1.5,
    "ADAPTIVE_MODE": True, 
    "ADAPTATION_WINDOW": 6,
    
    "BREAKOUT_DETECTION": True, 
    "BREAKOUT_THRESHOLD": 18, 
    "CLUSTER_DETECTION": True,
    "CLUSTER_WINDOW": 40, 
    "PATTERN_MEMORY": 60, 
    "DYNAMIC_TOP_NUMBERS": True,
    "MIN_TOP_NUMBERS": 12, 
    "MAX_TOP_NUMBERS": 20,
    
    "CONFIDENCE_BOOST_PER_100_DATA": 0.10,
}

# --- Configuración del Modo OUTSIDE ---
IA_CONFIG_OUTSIDE: Dict[str, Any] = {
    "MIN_HISTORY_FOR_ANALYSIS": 8,
    
    # ✅ MEJORA 2c: LONG_TERM_BALANCE_WEIGHT puesto en 0.0 (estrategia simplificada)
    "LOOKBACKS": {
        "long_term_balance": 75, # Ventana larga (DESACTIVADA)
        "momentum": 12           # Ventana corta (Momentum) - ESTRATEGIA PRINCIPAL
    },
    "WINDOW_WEIGHTS": {
        "long_term_balance": 0.0,  # ✅ MEJORA 2c: Cambiado de 40.0 a 0.0
        "momentum": 20.0
    },
    
    "ZERO_PROTECTION_WEIGHT": 10.0,
    "STREAK_ANALYSIS_WEIGHT": 15.0,
    "STREAK_ANALYSIS_WINDOW": 20,
    "STREAK_THRESHOLD": 0.6,
    "CHOP_THRESHOLD": 0.4,
    "MAX_GALE": 2,
    "COOLDOWN_ROUNDS": 2, # Unificado
    "FORCE_ENTRY": True
}


# ==============================================================================
# ESTADO GLOBAL DEL SERVIDOR
# ==============================================================================

# --- Estado Compartido ---
history: List[int] = []
game_state: str = "LEARNING" # LEARNING, IDLE, AWAITING_INITIAL_RESULT, AWAITING_G1_RESULT, AWAITING_G2_RESULT
active_bet: Optional[Dict[str, Any]] = None
current_mode: str = "neighbors" # 'neighbors' o 'outside'
cooldown_rounds: int = 0
new_update_event = asyncio.Event()
latest_data_for_frontend: Optional[Dict[str, Any]] = None

# --- Estado Específico de Neighbors ---
is_stable: bool = True
recent_pairs = deque([], maxlen=IA_CONFIG_NEIGHBORS["REPEAT_BUFFER_PAIRS"])
last_pair_used: Optional[tuple] = None
same_pair_streak = 0
pair_performance: Dict[tuple, Dict[str, Any]] = {}
number_last_seen: Dict[int, int] = {n: 0 for n in ROULETTE_ORDER}
pattern_memory: deque = deque(maxlen=IA_CONFIG_NEIGHBORS["PATTERN_MEMORY"])
cluster_cache: Dict[str, List[List[int]]] = {}
sector_momentum: Dict[str, float] = {}

# --- Estado Específico de Outside ---
zero_protection_active: bool = False

# --- ESTADÍSTICAS SEPARADAS (MEJORA v3.0) ---
stats: Dict[str, Dict[str, Any]] = {
    "neighbors": {
        "total_wins": 0, "total_losses": 0,
        "wins_initial": 0, "wins_g1": 0, "wins_g2": 0,
        "total_signals": 0, "current_win_streak": 0,
        "recent_results": deque(maxlen=IA_CONFIG_NEIGHBORS["ADAPTATION_WINDOW"])
    },
    "outside": {
        "total_wins": 0, "total_losses": 0,
        "wins_initial": 0, "wins_g1": 0, "wins_g2": 0,
        "total_signals": 0, "current_win_streak": 0,
        "recent_results": deque(maxlen=20)
    }
}

# ==============================================================================
# LÓGICA DE MODELOS (PYDANTIC)
# ==============================================================================

class NumberInput(BaseModel):
    number: int

class ModeInput(BaseModel):
    mode: str

# ==============================================================================
# FUNCIONES HELPER (NEIGHBORS)
# ==============================================================================

def get_current_phase() -> str:
    data_count = len(history)
    if data_count < IA_CONFIG_NEIGHBORS["LEARNING_PHASE_MIN"]:
        return "LEARNING"
    elif data_count < IA_CONFIG_NEIGHBORS["READY_PHASE_MIN"]:
        return "ANALYZING"
    elif data_count < IA_CONFIG_NEIGHBORS["OPTIMAL_DATA_THRESHOLD"]:
        return "READY"
    else:
        return "OPTIMAL"

def get_confidence_multiplier() -> float:
    data_count = len(history)
    if data_count < IA_CONFIG_NEIGHBORS["LEARNING_PHASE_MIN"]:
        return 0.0
    
    extra_data = data_count - IA_CONFIG_NEIGHBORS["LEARNING_PHASE_MIN"]
    boost = (extra_data // 100) * IA_CONFIG_NEIGHBORS["CONFIDENCE_BOOST_PER_100_DATA"]
    
    base_confidence = 0.65 if data_count < IA_CONFIG_NEIGHBORS["READY_PHASE_MIN"] else 1.0
    
    return min(base_confidence + boost, 2.0)

def get_neighbour_set(number: int, neighbours: Optional[int] = None) -> List[int]:
    if neighbours is None:
        neighbours = IA_CONFIG_NEIGHBORS.get("NEIGHBOURS", 3)
    if number not in ROULETTE_ORDER:
        return [number]
    idx = ROULETTE_ORDER.index(number)
    n = len(ROULETTE_ORDER)
    res = [number]
    for i in range(1, neighbours + 1):
        res.append(ROULETTE_ORDER[(idx - i) % n])
    for i in range(1, neighbours + 1):
        res.append(ROULETTE_ORDER[(idx + i) % n])
    return res

def get_wheel_distance(num1: int, num2: int) -> int:
    if num1 not in ROULETTE_ORDER or num2 not in ROULETTE_ORDER:
        return 0
    idx1 = ROULETTE_ORDER.index(num1)
    idx2 = ROULETTE_ORDER.index(num2)
    dist = abs(idx1 - idx2)
    return min(dist, len(ROULETTE_ORDER) - dist)

def update_number_tracking(number: int):
    for num in ROULETTE_ORDER:
        if num == number:
            number_last_seen[num] = 0
        else:
            number_last_seen[num] += 1

def detect_clusters(recent_history: List[int]) -> List[List[int]]:
    if not IA_CONFIG_NEIGHBORS["CLUSTER_DETECTION"] or len(recent_history) < IA_CONFIG_NEIGHBORS["CLUSTER_WINDOW"]:
        return []
    window = recent_history[-IA_CONFIG_NEIGHBORS["CLUSTER_WINDOW"]:]
    co_occurrence = defaultdict(lambda: defaultdict(int))
    for i in range(len(window) - 1):
        num1 = window[i]
        for j in range(i + 1, min(i + 4, len(window))):
            num2 = window[j]
            if num1 in ROULETTE_ORDER and num2 in ROULETTE_ORDER:
                co_occurrence[num1][num2] += 1
                co_occurrence[num2][num1] += 1
    clusters = []
    processed = set()
    for num1 in ROULETTE_ORDER:
        if num1 in processed: continue
        cluster = [num1]
        for num2, count in co_occurrence[num1].items():
            if count >= 3 and num2 not in processed:
                cluster.append(num2)
                processed.add(num2)
        if len(cluster) >= 3:
            clusters.append(cluster)
            processed.add(num1)
    return clusters

def detect_breakouts(history_window: List[int]) -> Dict[int, float]:
    if not IA_CONFIG_NEIGHBORS["BREAKOUT_DETECTION"]:
        return {n: 0.0 for n in ROULETTE_ORDER}
    breakout_scores = {}
    threshold = IA_CONFIG_NEIGHBORS["BREAKOUT_THRESHOLD"]
    avg_expected = len(history_window) / 37.0
    for num in ROULETTE_ORDER:
        rounds_without = number_last_seen.get(num, 0)
        if rounds_without >= threshold:
            base_score = ((rounds_without - threshold) / threshold) ** 1.5
            freq_count = history_window.count(num)
            if freq_count < avg_expected * 0.5:
                base_score *= 1.8
            breakout_scores[num] = min(base_score, 3.0)
        else:
            breakout_scores[num] = 0.0
    return breakout_scores

def analyze_sector_momentum(recent_history: List[int]) -> Dict[str, float]:
    if len(recent_history) < 30: return {}
    sectors = {
        'sector_1': ROULETTE_ORDER[0:9], 'sector_2': ROULETTE_ORDER[9:18],
        'sector_3': ROULETTE_ORDER[18:27], 'sector_4': ROULETTE_ORDER[27:37],
    }
    sector_hits_recent = {s: 0 for s in sectors}
    sector_hits_older = {s: 0 for s in sectors}
    window_recent = recent_history[-20:]
    window_older = recent_history[-40:-20] if len(recent_history) >= 40 else []
    for num in window_recent:
        for sector_name, sector_nums in sectors.items():
            if num in sector_nums: sector_hits_recent[sector_name] += 1
    for num in window_older:
        for sector_name, sector_nums in sectors.items():
            if num in sector_nums: sector_hits_older[sector_name] += 1
    momentum = {}
    for sector_name in sectors:
        recent = sector_hits_recent[sector_name]
        older = sector_hits_older[sector_name] if window_older else recent
        if older > 0: momentum[sector_name] = (recent - older) / older
        else: momentum[sector_name] = recent / 10.0
    return momentum

def get_sector_diversity_advanced(base_numbers: List[int]) -> float:
    if len(base_numbers) != 2: return 0.0
    num1, num2 = base_numbers
    dist = get_wheel_distance(num1, num2)
    optimal_dist = 9 
    distance_score = 1.0 if dist >= optimal_dist else 0.0
    idx1 = ROULETTE_ORDER.index(num1)
    idx2 = ROULETTE_ORDER.index(num2)
    half_bonus = 2.5 if (idx1 < 18) != (idx2 < 18) else 1.0
    quarter_bonus = 1.8 if abs(idx1 - idx2) > 9 else 1.0
    return distance_score * half_bonus * quarter_bonus

def analyze_coverage_quality(covered_numbers: List[int], analysis_data: Dict) -> Dict[str, float]:
    metrics = {'avg_frequency': 0, 'avg_recency': 0, 'avg_hot_zone': 0,
               'strong_numbers_count': 0, 'coverage_score': 0,
               'weak_numbers_count': 0, 'breakout_score': 0}
    hot_threshold = IA_CONFIG_NEIGHBORS["HOT_ZONE_THRESHOLD"]
    for num in covered_numbers:
        freq = analysis_data['frequency'].get(num, 0)
        recency = analysis_data['recency'].get(num, 0)
        hot = analysis_data['hot_zones'].get(num, 1.0)
        breakout = analysis_data.get('breakouts', {}).get(num, 0)
        strength = (freq * 4.0 + recency * 3.5 + hot * 4.5 + breakout * 2.0)
        if freq > 0.75 and hot > hot_threshold and recency > 0.4:
            metrics['strong_numbers_count'] += 1
        elif freq < 0.15 and hot < 0.8 and breakout < 0.5:
            metrics['weak_numbers_count'] += 1
        metrics['coverage_score'] += strength
        metrics['avg_frequency'] += freq
        metrics['avg_recency'] += recency
        metrics['avg_hot_zone'] += hot
        metrics['breakout_score'] += breakout
    count = len(covered_numbers)
    if count > 0:
        metrics['avg_frequency'] /= count
        metrics['avg_recency'] /= count
        metrics['avg_hot_zone'] /= count
        metrics['breakout_score'] /= count
        metrics['coverage_score'] /= count
        weak_ratio = metrics['weak_numbers_count'] / count
        metrics['coverage_score'] *= (1.0 - weak_ratio * 0.7)
    return metrics

def analyze_hot_zones(recent_history: List[int]) -> Dict[int, float]:
    if len(recent_history) < IA_CONFIG_NEIGHBORS["HOT_ZONE_WINDOW"]:
        return {n: 1.0 for n in ROULETTE_ORDER}
    window = recent_history[-IA_CONFIG_NEIGHBORS["HOT_ZONE_WINDOW"]:]
    zone_hits = {n: 0 for n in ROULETTE_ORDER}
    for i, num in enumerate(window):
        if num in ROULETTE_ORDER:
            weight = 0.7 + (i / len(window)) * 0.3
            neighbors = get_neighbour_set(num, 2)
            for n in neighbors:
                zone_hits[n] += weight
    avg_hits = sum(zone_hits.values()) / len(zone_hits) if zone_hits else 0
    hot_scores = {}
    for n, hits in zone_hits.items():
        if avg_hits > 0:
            score = hits / avg_hits
            hot_scores[n] = max(0.1, 1.0 + (score - 1.0) * 1.5)
        else:
            hot_scores[n] = 1.0
    return hot_scores

def analyze_distance_patterns(recent_history: List[int]) -> Dict[str, float]:
    window_size = IA_CONFIG_NEIGHBORS["STABILITY_WINDOW"]
    if len(recent_history) < window_size:
        return {"avg_distance": 0, "trend": 0, "volatility": 0.0, "is_stable": True}
    distances = []
    window = recent_history[-window_size:]
    for i in range(len(window) - 1):
        dist = get_wheel_distance(window[i], window[i+1])
        distances.append(dist)
    if not distances:
        return {"avg_distance": 0, "trend": 0, "volatility": 0.0, "is_stable": True}
    avg_dist = sum(distances) / len(distances)
    trend = 0
    if len(distances) >= 20:
        recent_avg = sum(distances[-10:]) / 10
        previous_avg = sum(distances[-20:-10]) / 10
        trend = recent_avg - previous_avg
    variance = sum((d - avg_dist) ** 2 for d in distances) / len(distances)
    volatility = variance ** 0.5
    is_stable = volatility <= IA_CONFIG_NEIGHBORS["VOLATILITY_THRESHOLD"]
    return {"avg_distance": avg_dist, "trend": trend, "volatility": volatility, "is_stable": is_stable}

def adapt_weights() -> Dict[str, float]:
    mode_stats = stats["neighbors"]
    if not IA_CONFIG_NEIGHBORS["ADAPTIVE_MODE"] or len(mode_stats["recent_results"]) < 4:
        return {}
    
    win_rate = sum(1 for r in mode_stats["recent_results"] if r) / len(mode_stats["recent_results"])
    confidence_mult = get_confidence_multiplier()
    adjustments = {}
    
    if win_rate < 0.80:
        adjustments = {
            "FREQUENCY_WEIGHT": 2.5 * confidence_mult, "HOT_ZONE_WEIGHT": 3.0 * confidence_mult,
            "RECENCY_WEIGHT": 2.2 * confidence_mult, "COVERAGE_QUALITY_WEIGHT": 2.5 * confidence_mult,
            "ACCURACY_WEIGHT": 1.8 * confidence_mult, "REPEAT_PENALTY": 3.0,
            "OVERLAP_PENALTY_MULTIPLIER": 3.5, "COLD_ZONE_PENALTY": 3.0,
            "DIVERSITY_WEIGHT": 0.3, "PATTERN_WEIGHT": 2.0 * confidence_mult,
        }
    elif win_rate < 0.90:
        adjustments = {
            "FREQUENCY_WEIGHT": 1.6 * confidence_mult, "HOT_ZONE_WEIGHT": 2.0 * confidence_mult,
            "RECENCY_WEIGHT": 1.5 * confidence_mult, "COVERAGE_QUALITY_WEIGHT": 1.8 * confidence_mult,
            "REPEAT_PENALTY": 2.2, "OVERLAP_PENALTY_MULTIPLIER": 2.5,
            "PATTERN_WEIGHT": 1.5 * confidence_mult,
        }
    else:
        adjustments = {"CONFIDENCE_MULTIPLIER": confidence_mult, "DIVERSITY_WEIGHT": 0.8, "COLD_ZONE_BONUS": 0.5}
    return adjustments

def get_pair_performance_score(pair: tuple) -> float:
    if pair not in pair_performance: return 1.0
    stats = pair_performance[pair]
    wins = stats.get('wins', 0)
    losses = stats.get('losses', 0)
    uses = stats.get('uses', 1)
    recent_wins = stats.get('recent_wins', 0)
    recent_uses = stats.get('recent_uses', 1)
    total_win_rate = (wins - losses * 0.5) / uses
    recent_win_rate = recent_wins / recent_uses
    combined_rate = (total_win_rate * 0.3) + (recent_win_rate * 0.7)
    if combined_rate > 0.85: return 2.5
    elif combined_rate > 0.70: return 1.8
    elif combined_rate < 0.40: return 0.2
    elif combined_rate < 0.55: return 0.6
    else: return 1.0

# ==============================================================================
# FUNCIONES HELPER (OUTSIDE)
# ==============================================================================

def get_number_properties(number: int) -> Dict[str, str]:
    """Obtiene las propiedades de un número para análisis"""
    if number == 0:
        return {"color": "GREEN", "parity": "ZERO", "range": "ZERO"}
    properties = {}
    if number in OUTSIDE_BETS["RED"]: properties["color"] = "RED"
    else: properties["color"] = "BLACK"
    if number in OUTSIDE_BETS["EVEN"]: properties["parity"] = "EVEN"
    else: properties["parity"] = "ODD"
    if number in OUTSIDE_BETS["LOW"]: properties["range"] = "LOW"
    else: properties["range"] = "HIGH"
    return properties

def analyze_trends(history_data: List[int]) -> Dict[str, Dict[str, float]]:
    """Analiza tendencias en las apuestas externas"""
    if len(history_data) < 5:
        return {}
    
    lookbacks = IA_CONFIG_OUTSIDE["LOOKBACKS"]
    trends = {}
    for window_name, window_size in lookbacks.items():
        window_data = history_data[-window_size:] if len(history_data) >= window_size else history_data
        window_trends = {}
        color_count = {"RED": 0, "BLACK": 0}
        parity_count = {"EVEN": 0, "ODD": 0}
        range_count = {"LOW": 0, "HIGH": 0}
        zero_count = 0
        for num in window_data:
            if num == 0:
                zero_count += 1
                continue
            props = get_number_properties(num)
            color_count[props["color"]] += 1
            parity_count[props["parity"]] += 1
            range_count[props["range"]] += 1
        total_non_zero = len(window_data) - zero_count
        if total_non_zero > 0:
            window_trends["RED"] = color_count["RED"] / total_non_zero
            window_trends["BLACK"] = color_count["BLACK"] / total_non_zero
            window_trends["EVEN"] = parity_count["EVEN"] / total_non_zero
            window_trends["ODD"] = parity_count["ODD"] / total_non_zero
            window_trends["LOW"] = range_count["LOW"] / total_non_zero
            window_trends["HIGH"] = range_count["HIGH"] / total_non_zero
            window_trends["ZERO_RATE"] = zero_count / len(window_data)
        trends[window_name] = window_trends
    return trends

def analyze_streaks(history_data: List[int], bet_type: str) -> Dict[str, float]:
    """
    Analiza rachas (streaks) y cortes (chops) para un tipo de apuesta.
    'bet_type' puede ser 'color', 'parity', o 'range'.
    """
    window_size = IA_CONFIG_OUTSIDE["STREAK_ANALYSIS_WINDOW"]
    if len(history_data) < window_size:
        return {}
    
    last_n = [get_number_properties(n) for n in history_data[-window_size:] if n != 0]
    if len(last_n) < 10:
        return {}

    streaks = 0
    chops = 0
    
    for i in range(len(last_n) - 1):
        prop_current = last_n[i].get(bet_type)
        prop_next = last_n[i+1].get(bet_type)
        
        if prop_current and prop_next:
            if prop_current == prop_next:
                streaks += 1
            else:
                chops += 1
    
    if streaks + chops == 0:
        return {}
    
    streak_ratio = streaks / (streaks + chops)
    last_prop = last_n[-1].get(bet_type)
    if not last_prop:
        return {}

    bonus_score = IA_CONFIG_OUTSIDE["STREAK_ANALYSIS_WEIGHT"]
    bonuses = {}

    if streak_ratio > IA_CONFIG_OUTSIDE["STREAK_THRESHOLD"]:
        if bet_type == "color":
            bonuses = {"RED": bonus_score if last_prop == "RED" else 0.0, "BLACK": bonus_score if last_prop == "BLACK" else 0.0}
        elif bet_type == "parity":
            bonuses = {"EVEN": bonus_score if last_prop == "EVEN" else 0.0, "ODD": bonus_score if last_prop == "ODD" else 0.0}
        elif bet_type == "range":
            bonuses = {"LOW": bonus_score if last_prop == "LOW" else 0.0, "HIGH": bonus_score if last_prop == "HIGH" else 0.0}
            
    elif streak_ratio < IA_CONFIG_OUTSIDE["CHOP_THRESHOLD"]:
        if bet_type == "color":
            bonuses = {"RED": bonus_score if last_prop == "BLACK" else 0.0, "BLACK": bonus_score if last_prop == "RED" else 0.0}
        elif bet_type == "parity":
            bonuses = {"EVEN": bonus_score if last_prop == "ODD" else 0.0, "ODD": bonus_score if last_prop == "EVEN" else 0.0}
        elif bet_type == "range":
            bonuses = {"LOW": bonus_score if last_prop == "HIGH" else 0.0, "HIGH": bonus_score if last_prop == "LOW" else 0.0}

    return bonuses


def calculate_bet_scores(history_data: List[int]) -> Dict[str, float]:
    """Calcula puntuaciones para apuestas externas, combinando tendencias y análisis de rachas."""
    if len(history_data) < 5:
        return {bet: 50.0 for bet in ["RED", "BLACK", "EVEN", "ODD", "LOW", "HIGH"]}

    trends = analyze_trends(history_data)
    window_weights = IA_CONFIG_OUTSIDE.get("WINDOW_WEIGHTS", {})
    scores = {bet: 0.0 for bet in ["RED", "BLACK", "EVEN", "ODD", "LOW", "HIGH"]}

    # 1. Análisis de Ventanas (MEJORA v3.8: Solo Momentum activo)
    for window_name, window_size in IA_CONFIG_OUTSIDE.get("LOOKBACKS", {}).items():
        if window_name not in trends: continue
        win_trends = trends[window_name]
        weight = window_weights.get(window_name, 1.0)
        
        # ✅ MEJORA 2c: Solo 'momentum' tiene peso (long_term_balance está en 0.0)
        contrarian = False  # Ya no usamos estrategia contrarian
        
        zero_rate = win_trends.get("ZERO_RATE", 0.0)
        sample_length = int(window_size)
        if len(history_data) < window_size:
            sample_length = len(history_data)
        total_non_zero = int(sample_length * (1 - zero_rate)) if sample_length > 0 else 0
        if total_non_zero <= 0: continue
            
        for bet_type in scores.keys():
            if bet_type not in win_trends: continue
            freq = win_trends[bet_type]
            diff = freq - 0.5
            z = diff * (total_non_zero ** 0.5)
            if contrarian:
                z = -z
            scores[bet_type] += z * weight

    # 2. Análisis de Rachas
    streak_bonuses = {}
    streak_bonuses.update(analyze_streaks(history_data, "color"))
    streak_bonuses.update(analyze_streaks(history_data, "parity"))
    streak_bonuses.update(analyze_streaks(history_data, "range"))

    for bet_type, bonus in streak_bonuses.items():
        if bet_type in scores:
            scores[bet_type] += bonus
            
    # 3. Protección contra Cero
    global zero_protection_active
    if history_data and history_data[-1] == 0:
        zero_protection_active = True
        zero_weight = IA_CONFIG_OUTSIDE.get("ZERO_PROTECTION_WEIGHT", 0.0)
        for bet_type in scores.keys():
            scores[bet_type] += zero_weight
    else:
        zero_protection_active = False

    return scores

# ==============================================================================
# LÓGICA DE PREDICCIÓN (MODOS)
# ==============================================================================

def find_best_number_bet() -> Dict[str, Any]:
    """Lógica de predicción para el modo NEIGHBORS"""
    global same_pair_streak, is_stable
    
    phase = get_current_phase()
    confidence_mult = get_confidence_multiplier()
    neighbours = IA_CONFIG_NEIGHBORS["NEIGHBOURS"]
    
    weight_adjustments = adapt_weights()
    
    def get_weight(key: str) -> float:
        base = IA_CONFIG_NEIGHBORS[key]
        adjustment = weight_adjustments.get(key, 1.0)
        if "CONFIDENCE_MULTIPLIER" in weight_adjustments:
            adjustment *= weight_adjustments["CONFIDENCE_MULTIPLIER"]
        return base * adjustment
    
    if len(history) < IA_CONFIG_NEIGHBORS["MIN_HISTORY_FOR_ANALYSIS"]:
        return {
            "base_numbers": [15, 32],
            "covered_numbers": sorted(set(get_neighbour_set(15, neighbours)) | set(get_neighbour_set(32, neighbours))),
            "coverage_count": 18,
            "reasoning": f"Fase: {phase} - Recolectando datos ({len(history)}/{IA_CONFIG_NEIGHBORS['LEARNING_PHASE_MIN']})",
            "phase": phase,
            "confidence": 0.0,
            "mode": "neighbors"
        }
    
    lookback = min(IA_CONFIG_NEIGHBORS["LOOKBACK"], len(history))
    tr = history[-lookback:]
    
    dfreq = {n: 0.0 for n in ROULETTE_ORDER}
    gamma = IA_CONFIG_NEIGHBORS["DECAY_GAMMA"]
    for k, num in enumerate(reversed(tr)):
        if num in dfreq: dfreq[num] += (gamma ** k)
    max_freq = max(dfreq.values()) if dfreq.values() else 1
    if max_freq > 0:
        for n in dfreq: dfreq[n] /= max_freq
    
    recency = {n: 1.0 / (1 + number_last_seen.get(n, lookback) ** 0.5) for n in ROULETTE_ORDER}
    
    hot_zones = analyze_hot_zones(tr)
    
    breakouts = detect_breakouts(tr)
    distance_patterns = analyze_distance_patterns(tr)
    clusters = detect_clusters(tr)
    sector_momentum = analyze_sector_momentum(tr)
    
    is_stable = distance_patterns["is_stable"]
    
    analysis_data = {'frequency': dfreq, 'recency': recency, 'hot_zones': hot_zones, 'breakouts': breakouts}
    
    neigh_sets = {n: set(get_neighbour_set(n, neighbours)) for n in ROULETTE_ORDER}
    
    if IA_CONFIG_NEIGHBORS["DYNAMIC_TOP_NUMBERS"]:
        data_ratio = min(len(history) / IA_CONFIG_NEIGHBORS["OPTIMAL_DATA_THRESHOLD"], 1.5)
        top_n = int(IA_CONFIG_NEIGHBORS["MIN_TOP_NUMBERS"] + (IA_CONFIG_NEIGHBORS["MAX_TOP_NUMBERS"] - IA_CONFIG_NEIGHBORS["MIN_TOP_NUMBERS"]) * data_ratio)
    else:
        top_n = 18
    
    def composite_score(num):
        return (dfreq[num] * 3.0 + recency[num] * 2.5 + hot_zones[num] * 3.5 + breakouts[num] * 1.5)
    
    top_numbers = sorted(ROULETTE_ORDER, key=composite_score, reverse=True)[:top_n]
    
    best_pair = None
    best_score = float('-inf')
    pair_scores: Dict[tuple, float] = {}
    
    for i in top_numbers:
        for j in top_numbers:
            if i >= j: continue
            set_i = neigh_sets[i]
            set_j = neigh_sets[j]
            
            if len(set_i & set_j) > 0: continue
            
            cov = set_i | set_j
            C = 18
            
            coverage_quality = analyze_coverage_quality(list(cov), analysis_data)
            
            coverage_score = get_weight("COVERAGE_WEIGHT") * C
            coverage_quality_score = get_weight("COVERAGE_QUALITY_WEIGHT") * coverage_quality['coverage_score'] * 20
            frequency_score = get_weight("FREQUENCY_WEIGHT") * (dfreq[i] + dfreq[j])
            recency_score = get_weight("RECENCY_WEIGHT") * (recency[i] + recency[j])
            hot_zone_score = get_weight("HOT_ZONE_WEIGHT") * (hot_zones[i] + hot_zones[j])
            performance_score = get_pair_performance_score((i, j)) * 4.0 * confidence_mult
            diversity_score = get_weight("DIVERSITY_WEIGHT") * get_sector_diversity_advanced([i, j]) * 8
            cluster_bonus = 0
            for cluster in clusters:
                if i in cluster and j in cluster: cluster_bonus = get_weight("CLUSTER_WEIGHT") * 10; break
                elif i in cluster or j in cluster: cluster_bonus = get_weight("CLUSTER_WEIGHT") * 5
            breakout_bonus = (breakouts[i] + breakouts[j]) * 8
            pattern_score = 0
            if distance_patterns["trend"] > 1.5 and get_wheel_distance(i, j) > distance_patterns["avg_distance"]: pattern_score = get_weight("PATTERN_WEIGHT") * 5
            elif distance_patterns["trend"] < -1.5 and get_wheel_distance(i, j) < distance_patterns["avg_distance"]: pattern_score = get_weight("PATTERN_WEIGHT") * 5
            sector_bonus = 0
            if sector_momentum:
                idx1 = ROULETTE_ORDER.index(i); idx2 = ROULETTE_ORDER.index(j)
                sector1 = f"sector_{(idx1 // 9) + 1}"; sector2 = f"sector_{(idx2 // 9) + 1}"
                sector_bonus = (sector_momentum.get(sector1, 0) + sector_momentum.get(sector2, 0)) * 5
            p3 = 1.0 - ((1.0 - (C/37.0)) ** 3)
            accuracy_score = get_weight("ACCURACY_WEIGHT") * p3 * 30
            sc = (coverage_score + coverage_quality_score + frequency_score + 
                  recency_score + hot_zone_score + diversity_score + 
                  pattern_score + accuracy_score + performance_score +
                  cluster_bonus + breakout_bonus + sector_bonus)
            
            current_pair_sorted = tuple(sorted((i, j)))
            if current_pair_sorted in recent_pairs: sc -= get_weight("REPEAT_PENALTY") * 25
            if last_pair_used == current_pair_sorted: sc -= 60.0
            if last_pair_used is not None:
                li, lj = last_pair_used
                prev_covered = neigh_sets[li] | neigh_sets[lj]
                inter = len(cov & prev_covered); uni = len(cov | prev_covered) or 1
                overlap_ratio = inter / uni
                sc -= overlap_ratio * get_weight("OVERLAP_PENALTY_MULTIPLIER") * 120.0
            base_cold_penalty = 0
            if hot_zones[i] < IA_CONFIG_NEIGHBORS["HOT_ZONE_THRESHOLD"] * 0.6: base_cold_penalty += get_weight("COLD_ZONE_PENALTY") * 3
            if hot_zones[j] < IA_CONFIG_NEIGHBORS["HOT_ZONE_THRESHOLD"] * 0.6: base_cold_penalty += get_weight("COLD_ZONE_PENALTY") * 3
            sc -= base_cold_penalty
            if len(history) > 200:
                if breakouts[i] < 0.5 and dfreq[i] < 0.3: sc -= 15.0
                if breakouts[j] < 0.5 and dfreq[j] < 0.3: sc -= 15.0
            
            pair_scores[(i, j)] = sc
            if sc > best_score:
                best_score = sc
                best_pair = (i, j)
    
    chosen = best_pair
    
    if chosen is None:
        print(f"⚠️ ALERTA: No se encontró par sin solapamiento en top_numbers (Top {top_n}). Buscando en TODOS los 37 números...")
        fallback_best_pair = None; fallback_best_score = float('-inf')
        all_numbers = ROULETTE_ORDER
        for i in all_numbers:
            for j in all_numbers:
                if i >= j: continue
                if len(neigh_sets[i] & neigh_sets[j]) > 0: continue
                sc = composite_score(i) + composite_score(j)
                sc += get_pair_performance_score(tuple(sorted((i,j))))
                if sc > fallback_best_score:
                    fallback_best_score = sc
                    fallback_best_pair = (i, j)
        if fallback_best_pair:
            base_nums = [fallback_best_pair[0], fallback_best_pair[1]]
            print(f"✅ Fallback exitoso: {base_nums} (encontrado en búsqueda completa)")
        else:
            print("🚨 ERROR CRÍTICO: No se encontró NINGÚN par sin solapamiento. Usando par de emergencia [0, 11].")
            base_nums = [0, 11]
    else:
        base_nums = [chosen[0], chosen[1]]
    
    current_pair_sorted = tuple(sorted(base_nums))
    if last_pair_used is not None and current_pair_sorted == last_pair_used:
        same_pair_streak += 1
    else:
        same_pair_streak = 0
    if same_pair_streak >= IA_CONFIG_NEIGHBORS["MAX_CONSEC_SAME_PAIR"]:
        alternatives = []
        for (p, sc) in pair_scores.items():
            if set(p) == set(base_nums): continue
            if len(neigh_sets[p[0]] & neigh_sets[p[1]]) > 0: continue
            current_cov = neigh_sets[base_nums[0]] | neigh_sets[base_nums[1]]
            alt_cov = neigh_sets[p[0]] | neigh_sets[p[1]]
            overlap = len(current_cov & alt_cov)
            if overlap < 6:
                alternatives.append((p, sc, overlap))
        if alternatives:
            alternatives.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            base_nums = [alternatives[0][0][0], alternatives[0][0][1]]
            same_pair_streak = 0
            print("🔄 CAMBIO ESTRATÉGICO DE PAR")

    covered = neigh_sets[base_nums[0]] | neigh_sets[base_nums[1]]
    C = len(covered)
    
    p3 = 1.0 - ((1.0 - (C/37.0)) ** 3)
    prediction_confidence = min(p3 * confidence_mult * 100, 99.9)
    final_quality = analyze_coverage_quality(list(covered), analysis_data)
    
    volatility_info = f"Volat={distance_patterns['volatility']:.2f}"
    
    reasoning = (
        f"Fase:{phase} | Data:{len(history)} | C={C} | P3={p3*100:.1f}% | "
        f"Conf={prediction_confidence:.1f}% | {volatility_info} | "
        f"Qual={final_quality['coverage_score']:.2f} | "
        f"Strong={final_quality['strong_numbers_count']}/{C}"
    )
    
    return {
        "mode": "neighbors",
        "base_numbers": base_nums,
        "covered_numbers": sorted(covered),
        "coverage_count": C,
        "reasoning": reasoning,
        "phase": phase,
        "confidence": prediction_confidence,
        "quality_metrics": final_quality,
        "distance_patterns": distance_patterns
    }

def find_best_outside_bet() -> Dict[str, Any]:
    """Lógica de predicción para el modo OUTSIDE"""
    if len(history) < IA_CONFIG_OUTSIDE["MIN_HISTORY_FOR_ANALYSIS"]:
        # ✅ MEJORA 2b: Ya NO se añade el '0' como seguro
        covered_nums_set_fallback = set(OUTSIDE_BETS["RED"])
        
        return {
            "mode": "outside",
            "bet_type": "RED",
            "covered_numbers": sorted(list(covered_nums_set_fallback)),
            "coverage_count": 18,  # Ya no es 19
            "reasoning": f"Recolectando datos ({len(history)}/{IA_CONFIG_OUTSIDE['MIN_HISTORY_FOR_ANALYSIS']})",
            "score": 0.0
        }
    
    scores = calculate_bet_scores(history)
    
    best_bet = max(scores, key=scores.get)
    best_score = scores[best_bet]
    
    print(f"[DEBUG-OUTSIDE] Scores: {scores}")
    print(f"[DEBUG-OUTSIDE] Mejor Apuesta: {best_bet} (Score: {best_score:.1f})")

    # ✅ MEJORA 2b: Ya NO se añade el '0' como seguro
    covered_nums_set = set(OUTSIDE_BETS[best_bet])
    
    return {
        "mode": "outside",
        "bet_type": best_bet,
        "covered_numbers": sorted(list(covered_nums_set)),
        "coverage_count": 18,  # Ya no es 19
        "score": best_score,
        "all_scores": scores,
        "reasoning": f"Análisis de tendencias v3.8 - Score: {best_score:.1f}"
    }

# ==============================================================================
# LÓGICA DE ESTADO Y ESTADÍSTICAS
# ==============================================================================

def get_all_stats() -> Dict[str, Any]:
    """Obtiene un diccionario con las estadísticas de AMBOS modos + estado global."""
    all_stats = {}
    
    for mode in ["neighbors", "outside"]:
        s = stats[mode]
        cycles = s["total_wins"] + s["total_losses"]
        acc = (s["total_wins"] / cycles * 100) if cycles > 0 else 0.0
        
        recent_acc = 0.0
        if len(s["recent_results"]) > 0:
            recent_acc = sum(1 for r in s["recent_results"] if r) / len(s["recent_results"]) * 100
        
        all_stats[mode] = {
            "wins": s["total_wins"],
            "losses": s["total_losses"],
            "cycles": cycles,
            "wins_initial": s["wins_initial"],
            "wins_g1": s["wins_g1"],
            "wins_g2": s["wins_g2"],
            "total_signals": s["total_signals"],
            "accuracy_percentage": round(acc, 1),
            "recent_accuracy": round(recent_acc, 1),
            "current_win_streak": s["current_win_streak"],
        }
    
    current_phase = get_current_phase()
    
    all_stats["global"] = {
        "phase": current_phase,
        "data_count": len(history),
        "confidence_multiplier": round(get_confidence_multiplier(), 2),
        "ready_to_bet": len(history) >= IA_CONFIG_NEIGHBORS["LEARNING_PHASE_MIN"],
        "is_stable": is_stable,
        "current_mode": current_mode,
        "game_state": game_state,
        "cooldown_rounds": cooldown_rounds,
    }
    return all_stats


# ==============================================================================
# ENDPOINTS DE FASTAPI
# ==============================================================================

@app.post("/process_number")
async def process_number(data: NumberInput):
    global game_state, active_bet, history, latest_data_for_frontend, cooldown_rounds, recent_pairs, last_pair_used, same_pair_streak, pair_performance, is_stable, zero_protection_active
    
    number = data.number
    if not (0 <= number <= 36):
        raise HTTPException(status_code=400, detail="Número inválido.")
    
    history.append(number)
    
    if current_mode == "neighbors":
         update_number_tracking(number)
    
    current_phase = get_current_phase()
    response: Dict[str, Any] = {"action": "WAIT"}
    bet_active = (game_state not in ["IDLE", "LEARNING"])
    
    min_learn_phase = IA_CONFIG_NEIGHBORS["LEARNING_PHASE_MIN"]
    if current_phase == "LEARNING" and len(history) < min_learn_phase:
        if not bet_active:
            response = {
                "action": "LEARNING",
                "message": f"Recolectando datos: {len(history)}/{min_learn_phase}",
                "progress": (len(history) / min_learn_phase) * 100,
                "phase": current_phase,
            }
    
    if len(history) >= min_learn_phase and game_state == "LEARNING":
        game_state = "IDLE"
        cooldown_rounds = 0
        print(f"✅ SISTEMA ACTIVADO - {len(history)} datos recopilados - Comenzando apuestas")
    
    elif bet_active:
        
        is_win = number in active_bet["covered_numbers"]

        bet_mode = active_bet.get("mode", "neighbors") 
        mode_stats = stats[bet_mode]
        
        mode_stats["recent_results"].append(is_win)
        
        if bet_mode == "neighbors" and "base_numbers" in active_bet:
            bi, bj = active_bet["base_numbers"]
            pair_key = tuple(sorted((bi, bj)))
            if pair_key not in pair_performance:
                pair_performance[pair_key] = {'wins': 0, 'losses': 0, 'uses': 0, 'recent_wins': 0, 'recent_uses': 0}
            pair_performance[pair_key]['uses'] += 1
            pair_performance[pair_key]['recent_uses'] += 1
            if is_win:
                pair_performance[pair_key]['wins'] += 1
                pair_performance[pair_key]['recent_wins'] += 1
            if pair_performance[pair_key]['recent_uses'] >= 20:
                pair_performance[pair_key]['recent_wins'] = 0
                pair_performance[pair_key]['recent_uses'] = 0
        
        if is_win:
            mode_stats["total_wins"] += 1
            mode_stats["current_win_streak"] += 1
            
            if game_state == "AWAITING_INITIAL_RESULT":
                mode_stats["wins_initial"] += 1; win_level = "INICIAL"
            elif game_state == "AWAITING_G1_RESULT":
                mode_stats["wins_g1"] += 1; win_level = "G1"
            else:
                mode_stats["wins_g2"] += 1; win_level = "G2"
            
            response = {
                "action": "WIN",
                "correct_number": number,
                "win_level": win_level,
                "bet_details": active_bet
            }
            
            if bet_mode == "neighbors" and "base_numbers" in active_bet:
                bi, bj = active_bet["base_numbers"]
                recent_pairs.append(tuple(sorted((bi, bj))))
                last_pair_used = tuple(sorted((bi, bj)))
                same_pair_streak = 0
            
            game_state = "IDLE"
            active_bet = None
            cooldown_rounds = IA_CONFIG_NEIGHBORS["COOLDOWN_ROUNDS"]
            print(f"✅ [{bet_mode.upper()}] VICTORIA! Nro: {number} | Nivel: {win_level} | Racha: {mode_stats['current_win_streak']}")
        
        else:
            if game_state == "AWAITING_INITIAL_RESULT":
                game_state = "AWAITING_G1_RESULT"
                response = {"action": "G1", "bet_details": active_bet}
            
            elif game_state == "AWAITING_G1_RESULT":
                game_state = "AWAITING_G2_RESULT"
                response = {"action": "G2", "bet_details": active_bet}
            
            else:
                mode_stats["total_losses"] += 1
                mode_stats["current_win_streak"] = 0
                
                if bet_mode == "neighbors" and "base_numbers" in active_bet:
                    bi, bj = active_bet["base_numbers"]
                    pair_key = tuple(sorted((bi, bj)))
                    if pair_key in pair_performance:
                        pair_performance[pair_key]['losses'] = pair_performance[pair_key].get('losses', 0) + 1
                        pair_performance[pair_key]['recent_uses'] = max(pair_performance[pair_key].get('recent_uses', 1), 5)
                        pair_performance[pair_key]['recent_wins'] = 0
                    recent_pairs.append(pair_key)
                    last_pair_used = pair_key
                
                response = {"action": "LOSS", "correct_number": number, "bet_details": active_bet}
                
                game_state = "IDLE"
                active_bet = None
                cooldown_rounds = IA_CONFIG_NEIGHBORS["COOLDOWN_ROUNDS"]
                print(f"❌ [{bet_mode.upper()}] PÉRDIDA. Nro: {number} | Racha reiniciada.")
    
    elif cooldown_rounds > 0:
        cooldown_rounds -= 1
        response = (
            {"action": "READY", "message": "Cooldown terminado"}
            if cooldown_rounds == 0
            else {"action": "COOLDOWN", "rounds_left": cooldown_rounds}
        )
    
    should_generate_signal = (
        game_state == "IDLE" and 
        cooldown_rounds == 0 and 
        len(history) >= min_learn_phase and
        response.get("action") not in ["WIN", "LOSS", "G1", "G2"]
    )
    
    if should_generate_signal:
        
        if current_mode == "neighbors":
            if IA_CONFIG_NEIGHBORS["FORCE_ENTRY"]:
                reco = find_best_number_bet()
                current_volatility = reco.get("distance_patterns", {}).get("volatility", 0.0)
                
                if IA_CONFIG_NEIGHBORS["USE_STABILITY_FILTER"] and not is_stable:
                    game_state = "IDLE"
                    active_bet = None
                    response = {
                        "action": "WAIT_STABILITY",
                        "message": f"PAUSA: Volatilidad alta ({current_volatility:.2f} > {IA_CONFIG_NEIGHBORS['VOLATILITY_THRESHOLD']})",
                        "reasoning": reco["reasoning"],
                        "volatility": current_volatility
                    }
                else:
                    game_state = "AWAITING_INITIAL_RESULT"
                    active_bet = reco
                    stats["neighbors"]["total_signals"] += 1
                    response = {
                        "action": "INITIAL",
                        "bet_details": reco,
                        "reasoning": reco["reasoning"],
                    }
                    stability_msg = "ESTABLE" if is_stable else f"INESTABLE ({current_volatility:.2f})"
                    print(f"🎯 [NEIGHBORS] NUEVA APUESTA - Par: {reco['base_numbers']} | Cobertura: {reco['coverage_count']} | Conf: {reco['confidence']:.1f}% | Est: {stability_msg}")

        elif current_mode == "outside":
            if IA_CONFIG_OUTSIDE["FORCE_ENTRY"]:
                if len(history) < IA_CONFIG_OUTSIDE["MIN_HISTORY_FOR_ANALYSIS"]:
                    response = {
                        "action": "WAIT_DATA_OUTSIDE",
                        "message": f"Datos insuficientes para modo Outside ({len(history)}/{IA_CONFIG_OUTSIDE['MIN_HISTORY_FOR_ANALYSIS']})"
                    }
                else:
                    reco = find_best_outside_bet()
                    game_state = "AWAITING_INITIAL_RESULT"
                    active_bet = reco
                    stats["outside"]["total_signals"] += 1
                    response = {
                        "action": "INITIAL",
                        "bet_details": reco,
                        "reasoning": reco["reasoning"],
                    }
                    print(f"🎯 [OUTSIDE] NUEVA APUESTA - Tipo: {reco['bet_type']} | Cobertura: {reco['coverage_count']} | Score: {reco['score']:.1f}")
    
    response["stats"] = get_all_stats()
    response["history"] = history[-20:]
    response["last_number"] = number
    
    latest_data_for_frontend = response
    new_update_event.set()
    new_update_event.clear()
    
    return response

@app.post("/set_mode")
async def set_mode(data: ModeInput):
    global current_mode, game_state, active_bet, cooldown_rounds
    
    if data.mode not in ["neighbors", "outside"]:
        raise HTTPException(status_code=400, detail="Modo inválido. Usar 'neighbors' o 'outside'.")
    
    if data.mode == current_mode:
        return {"status": "success", "message": f"El modo ya es {current_mode}", "stats": get_all_stats()}

    game_state = "IDLE"
    active_bet = None
    cooldown_rounds = 0
    current_mode = data.mode
    
    print(f"🔄 MODO CAMBIADO A: {current_mode.upper()}")
    
    response = {
        "action": "MODE_CHANGE",
        "message": f"Modo cambiado a {current_mode}",
        "stats": get_all_stats(),
        "history": history[-20:],
        "last_number": history[-1] if history else None
    }
    
    global latest_data_for_frontend, new_update_event
    latest_data_for_frontend = response
    new_update_event.set()
    new_update_event.clear()
    
    return {"status": "success", "message": f"Modo cambiado a {current_mode}", "stats": get_all_stats()}


@app.get("/listen_for_updates")
async def listen_for_updates():
    """Endpoint de long-polling para el frontend."""
    try:
        await asyncio.wait_for(new_update_event.wait(), timeout=80.0)
        if latest_data_for_frontend is None:
            return {"action": "NO_UPDATE"}
        return latest_data_for_frontend
    except asyncio.TimeoutError:
        return {"action": "NO_UPDATE"}
    except Exception as e:
        return {"action": "ERROR", "message": str(e)}

@app.get("/initial_state")
def get_initial_state():
    """Obtiene el estado completo al cargar la página."""
    try:
        return {
            "history": history[-20:] if history else [],
            "stats": get_all_stats()
        }
    except Exception as e:
        print(f"Error en initial_state: {e}")
        history.clear()
        reset_all_internal()
        return {
            "history": [],
            "stats": get_all_stats()
        }

@app.post("/delete_last_number")
def delete_last_number():
    global game_state, active_bet, history, cooldown_rounds, recent_pairs, last_pair_used, same_pair_streak, number_last_seen
    if not history:
        return {"status": "no history to delete"}
    
    removed = history.pop()
    
    temp_history_for_tracking = list(history)
    
    number_last_seen = {n: 0 for n in ROULETTE_ORDER}
    history.clear() 
    
    for num in temp_history_for_tracking:
        history.append(num)
        
        for n_internal in ROULETTE_ORDER:
            if n_internal == num:
                number_last_seen[n_internal] = 0
            else:
                number_last_seen[n_internal] += 1
    
    game_state = "LEARNING" if len(history) < IA_CONFIG_NEIGHBORS["LEARNING_PHASE_MIN"] else "IDLE"
    active_bet = None
    cooldown_rounds = 0
    same_pair_streak = 0
    last_pair_used = None
    
    print(f"🗑️ Número eliminado: {removed} | Quedan: {len(history)} | Estado reseteado a IDLE/LEARNING")
    
    return {
        "status": "success",
        "new_history": history[-20:],
        "stats": get_all_stats()
    }


def reset_stats_internal():
    """Función interna para reiniciar estadísticas de ambos modos."""
    global stats, pair_performance, last_pair_used, same_pair_streak
    
    for mode in stats.keys():
        stats[mode]["total_wins"] = 0
        stats[mode]["total_losses"] = 0
        stats[mode]["wins_initial"] = 0
        stats[mode]["wins_g1"] = 0
        stats[mode]["wins_g2"] = 0
        stats[mode]["total_signals"] = 0
        stats[mode]["current_win_streak"] = 0
        stats[mode]["recent_results"].clear()
    
    pair_performance.clear()
    last_pair_used = None
    same_pair_streak = 0
    
@app.post("/reset_stats")
def reset_stats():
    """Resetea contadores de GALE para ambos modos, pero mantiene el historial."""
    global game_state, active_bet, cooldown_rounds
    
    reset_stats_internal()
    
    game_state = "LEARNING" if len(history) < IA_CONFIG_NEIGHBORS["LEARNING_PHASE_MIN"] else "IDLE"
    active_bet = None
    cooldown_rounds = 0
    
    print("🔄 Estadísticas (ambos modos) reiniciadas")
    return {"status": "success", "stats": get_all_stats()}

def reset_all_internal():
    """Función interna para reseteo completo."""
    global history, game_state, active_bet, cooldown_rounds, number_last_seen, pattern_memory, cluster_cache
    
    history.clear()
    reset_stats_internal()
    
    recent_pairs.clear()
    pattern_memory.clear()
    cluster_cache.clear()
    number_last_seen = {n: 0 for n in ROULETTE_ORDER}
    
    game_state = "LEARNING"
    active_bet = None
    cooldown_rounds = 0

@app.post("/reset_all")
def reset_all():
    """Resetea historial, estadísticas y todo el estado del sistema."""
    reset_all_internal()
    print("🔄 RESET COMPLETO - Sistema reiniciado a fase de aprendizaje")
    return {"status": "success", "stats": get_all_stats(), "message": "Sistema reiniciado completamente"}

@app.get("/debug_info")
def get_debug_info():
    """Muestra información de debug para el MODO ACTUAL."""
    
    if len(history) < 5:
        return {
            "message": f"Fase de aprendizaje: {len(history)}/{IA_CONFIG_NEIGHBORS['LEARNING_PHASE_MIN']} datos",
            "current_mode": current_mode,
            "ready": False
        }
    
    if current_mode == "neighbors":
        reco = find_best_number_bet()
        tr = history[-IA_CONFIG_NEIGHBORS["LOOKBACK"]:]
        hot_zones = analyze_hot_zones(tr)
        breakouts = detect_breakouts(tr)
        clusters = detect_clusters(tr)
        top_pairs = sorted(pair_performance.items(), key=lambda x: get_pair_performance_score(x[0]), reverse=True)[:5]
        
        return {
            "current_mode": "neighbors",
            "current_recommendation": reco,
            "history_length": len(history),
            "game_state": game_state,
            "phase": get_current_phase(),
            "confidence_multiplier": get_confidence_multiplier(),
            "top_hot_zones": sorted(hot_zones.items(), key=lambda x: x[1], reverse=True)[:5],
            "top_breakout_candidates": sorted(breakouts.items(), key=lambda x: x[1], reverse=True)[:5],
            "active_clusters": clusters[:3] if clusters else [],
            "top_performing_pairs": [{"pair": p, "score": get_pair_performance_score(p), "stats": s} for p, s in top_pairs],
            "recent_win_rate": stats["neighbors"]["recent_accuracy"],
            "current_adaptive_weights": adapt_weights(),
            "current_stability": reco.get("distance_patterns", {}),
        }
    
    else:
        scores = calculate_bet_scores(history)
        trends = analyze_trends(history)
        reco = find_best_outside_bet()
        
        return {
            "current_mode": "outside",
            "bet_scores": scores,
            "trend_analysis": trends,
            "streak_analysis_color": analyze_streaks(history, "color"),
            "streak_analysis_parity": analyze_streaks(history, "parity"),
            "streak_analysis_range": analyze_streaks(history, "range"),
            "current_recommendation": reco,
            "history_length": len(history),
            "game_state": game_state,
            "zero_protection": zero_protection_active,
            "config": IA_CONFIG_OUTSIDE
        }


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 SERVIDOR IA COMBINADO v3.8 (Optimizado para Acertividad)")
    print("=" * 70)
    print(f"   ✓ ¡NUEVO! Sirviendo HTML en http://127.0.0.1:8000/")
    print("=" * 70)
    print("   MEJORAS v3.8:")
    print("     ✅ DIVERSITY_WEIGHT: 0.4 → 4.0 (ahora tiene impacto real)")
    print("     ✅ Eliminado '0' como seguro en Outside (EV corregido)")
    print("     ✅ Estrategia Outside simplificada (solo Momentum)")
    print("=" * 70)
    print("   MODO 1: NEIGHBORS (Ultra-Optimizado v2.5 Lógica)")
    print(f"     ✓ Fase 1 (0-{IA_CONFIG_NEIGHBORS['LEARNING_PHASE_MIN']}): APRENDIZAJE")
    print(f"     ✓ Fase 2 ({IA_CONFIG_NEIGHBORS['LEARNING_PHASE_MIN']}-{IA_CONFIG_NEIGHBORS['READY_PHASE_MIN']}): ANÁLISIS")
    print(f"     ✓ Fase 3 ({IA_CONFIG_NEIGHBORS['READY_PHASE_MIN']}+): ÓPTIMO")
    print("=" * 70)
    print("   MODO 2: OUTSIDE (Apuestas Externas)")
    print(f"     ✓ Análisis de Rachas y Cortes")
    print(f"     ✓ Estrategia Momentum pura (sin contradicciones)")
    print("=" * 70)
    print("   ✓ Estadísticas GALE separadas por modo.")
    print(f"   ✓ Cooldown unificado: {IA_CONFIG_NEIGHBORS['COOLDOWN_ROUNDS']} rondas.")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000)