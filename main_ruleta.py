# -*- coding: utf-8 -*-
"""
Backend Combinado de IA de Ruleta v3.8
=======================================
(Archivo modificado por Gemini para el despliegue en Railway
 CON TAREAS EN SEGUNDO PLANO para arreglar timeouts)
"""

# --- IMPORTS (CON ADICIONES PARA RAILWAY) ---
import uvicorn
import asyncio
import math
import random
import os
import logging # <-- Añadido para logs de depuración
from pathlib import Path # <-- Añadido para rutas absolutas
from typing import List, Dict, Any, Optional, Tuple
from collections import deque, Counter, defaultdict
# --- ¡IMPORTANTE! Añadido BackgroundTasks ---
from fastapi import FastAPI, HTTPException, BackgroundTasks
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

# --- CONFIGURACIÓN DE LOGGING Y RUTA BASE (AÑADIDO PARA RAILWAY) ---

# Configura un logger para que podamos ver los mensajes en los logs de Railway
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ¡NUEVO! Define el directorio base de tu proyecto ---
# Esto crea una ruta absoluta al directorio donde se encuentra este script
BASE_DIR = Path(__file__).resolve().parent

# --- FIN: ENDPOINT PARA SERVIR HTML ---


# --- ¡MODIFICADO! Esta es la ruta que da el HTML ---
@app.get("/", response_class=HTMLResponse)
async def get_root():
    """
    Sirve el archivo index.html principal.
    (Versión corregida para Railway)
    """
    
    # --- ¡CORREGIDO! Apunta a 'index.html', el archivo que subiste
    html_file_path = BASE_DIR / "index.html"
    
    logger.info(f"Petición a /: Intentando servir archivo desde: {html_file_path}")
    
    try:
        # Abre el archivo con codificación utf-8 (buena práctica)
        with open(html_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        logger.info("Petición a /: ¡ÉXITO! index.html leído y servido.")
        # --- ¡CORREGIDO! Devuelve 200 OK (vital para el health check)
        return HTMLResponse(content=content, status_code=200)

    except FileNotFoundError:
        logger.error(f"¡¡¡ERROR CRÍTICO!!! No se encontró index.html en la ruta: {html_file_path}")
        # Devuelve un error 500 para que lo veas en el navegador si algo sale mal
        return HTMLResponse(
            content=f"Error 500: FileNotFoundError. No se pudo encontrar 'index.html' en la ruta esperada: {html_file_path}",
            status_code=500
        )
    
    except Exception as e:
        logger.error(f"¡¡¡ERROR CRÍTICO!!! Error inesperado al leer index.html: {e}")
        # Esto te mostrará cualquier otro error
        return HTMLResponse(
            content=f"Error 500: Error inesperado del servidor. {e}",
            status_code=500
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
    "LEARNING_PHASE_MIN": 195,
    "READY_PHASE_MIN": 190,
    "OPTIMAL_DATA_THRESHOLD": 190,
    "MIN_HISTORY_FOR_ANALYSIS": 50,
    "LOOKBACK": 190,
    "MAX_GALE": 2,
    "COOLDOWN_ROUNDS": 5,
    "FORCE_ENTRY": True,
    "NEIGHBOURS": 4,
    "USE_STABILITY_FILTER": True,
    "STABILITY_WINDOW": 30,
    "VOLATILITY_THRESHOLD": 9.5,
    "COVERAGE_WEIGHT": 2.0, 
    "COVERAGE_QUALITY_WEIGHT": 10.0, 
    "FREQUENCY_WEIGHT": 8.0,
    "RECENCY_WEIGHT": 7.0, 
    "HOT_ZONE_WEIGHT": 9.0, 
    "DIVERSITY_WEIGHT": 4.0,
    "COLD_ZONE_BONUS": 0.1, 
    "PATTERN_WEIGHT": 5.0, 
    "CLUSTER_WEIGHT": 6.0,
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
    "LOOKBACKS": {
        "long_term_balance": 75,
        "momentum": 12
    },
    "WINDOW_WEIGHTS": {
        "long_term_balance": 0.0,
        "momentum": 20.0
    },
    "ZERO_PROTECTION_WEIGHT": 10.0,
    "STREAK_ANALYSIS_WEIGHT": 15.0,
    "STREAK_ANALYSIS_WINDOW": 20,
    "STREAK_THRESHOLD": 0.6,
    "CHOP_THRESHOLD": 0.4,
    "MAX_GALE": 2,
    "COOLDOWN_ROUNDS": 2,
    "FORCE_ENTRY": True
}


# ==============================================================================
# ESTADO GLOBAL DEL SERVIDOR
# ==============================================================================

history: List[int] = []
# --- ¡NUEVO ESTADO! Para evitar atascos
game_state: str = "LEARNING" # LEARNING, IDLE, AWAITING_..., ANALYZING_BG
active_bet: Optional[Dict[str, Any]] = None
current_mode: str = "neighbors"
cooldown_rounds: int = 0
new_update_event = asyncio.Event()
latest_data_for_frontend: Optional[Dict[str, Any]] = None
is_stable: bool = True
recent_pairs = deque([], maxlen=IA_CONFIG_NEIGHBORS["REPEAT_BUFFER_PAIRS"])
last_pair_used: Optional[tuple] = None
same_pair_streak = 0
pair_performance: Dict[tuple, Dict[str, Any]] = {}
number_last_seen: Dict[int, int] = {n: 0 for n in ROULETTE_ORDER}
pattern_memory: deque = deque(maxlen=IA_CONFIG_NEIGHBORS["PATTERN_MEMORY"])
cluster_cache: Dict[str, List[List[int]]] = {}
sector_momentum: Dict[str, float] = {}
zero_protection_active: bool = False

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
    
# --- ¡NUEVO! Modelo para carga masiva ---
class BulkNumbersInput(BaseModel):
    numbers: List[int]

# ==============================================================================
# FUNCIONES HELPER (NEIGHBORS)
# (Todo tu código de IA va aquí, sin cambios)
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
# (Tu código de IA)
# ==============================================================================

def get_number_properties(number: int) -> Dict[str, str]:
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
        # Racha: apostar a que sigue
        if bet_type == "color":
            bonuses = {"RED": bonus_score if last_prop == "RED" else 0.0, "BLACK": bonus_score if last_prop == "BLACK" else 0.0}
        elif bet_type == "parity":
            bonuses = {"EVEN": bonus_score if last_prop == "EVEN" else 0.0, "ODD": bonus_score if last_prop == "ODD" else 0.0}
        elif bet_type == "range":
            bonuses = {"LOW": bonus_score if last_prop == "LOW" else 0.0, "HIGH": bonus_score if last_prop == "HIGH" else 0.0}

    elif streak_ratio < IA_CONFIG_OUTSIDE["CHOP_THRESHOLD"]:
        # Corte: apostar a que cambia
        if bet_type == "color":
            bonuses = {"RED": bonus_score if last_prop == "BLACK" else 0.0, "BLACK": bonus_score if last_prop == "RED" else 0.0}
        elif bet_type == "parity":
            bonuses = {"EVEN": bonus_score if last_prop == "ODD" else 0.0, "ODD": bonus_score if last_prop == "EVEN" else 0.0}
        elif bet_type == "range":
            bonuses = {"LOW": bonus_score if last_prop == "HIGH" else 0.0, "HIGH": bonus_score if last_prop == "LOW" else 0.0}
            
    return bonuses


def find_best_outside_bet() -> Dict[str, Any]:
    """Lógica de predicción para el modo OUTSIDE"""
    global zero_protection_active
    history_data = history[-100:]
    if len(history_data) < IA_CONFIG_OUTSIDE["MIN_HISTORY_FOR_ANALYSIS"]:
        return {"bet": "LEARNING", "covered_numbers": [], "gale_level": 0, "reasoning": "Datos insuficientes"}

    scores = {"RED": 0.0, "BLACK": 0.0, "EVEN": 0.0, "ODD": 0.0, "LOW": 0.0, "HIGH": 0.0}
    trends_by_window = analyze_trends(history_data)
    
    for window_name, window_trends in trends_by_window.items():
        weight = IA_CONFIG_OUTSIDE["WINDOW_WEIGHTS"].get(window_name, 0.0)
        if weight == 0.0:
            continue
            
        contrarian = (window_name == "long_term_balance")
        
        for bet_type in scores.keys():
            if bet_type not in window_trends: continue
            
            freq = window_trends[bet_type]
            diff = freq - 0.5
            total_non_zero = int(len(history_data) * (1.0 - window_trends.get("ZERO_RATE", 0.0)))
            if total_non_zero > 0:
                z = diff * (total_non_zero ** 0.5)
                if contrarian:
                    z = -z
                scores[bet_type] += z * weight

    streak_bonuses = {}
    streak_bonuses.update(analyze_streaks(history_data, "color"))
    streak_bonuses.update(analyze_streaks(history_data, "parity"))
    streak_bonuses.update(analyze_streaks(history_data, "range"))

    for bet_type, bonus in streak_bonuses.items():
        if bet_type in scores:
            scores[bet_type] += bonus
            
    global zero_protection_active
    if history_data and history_data[-1] == 0:
        zero_protection_active = True
        zero_weight = IA_CONFIG_OUTSIDE.get("ZERO_PROTECTION_WEIGHT", 0.0)
        for bet_type in scores.keys():
             scores[bet_type] += zero_weight
    else:
        zero_protection_active = False
        
    if not scores:
        return {"bet": "IDLE", "covered_numbers": [], "gale_level": 0, "reasoning": "No se generaron scores."}

    best_bet = max(scores, key=scores.get)
    best_score = scores[best_bet]

    if best_score < 0.1:
        return {"bet": "IDLE", "covered_numbers": [], "gale_level": 0, "reasoning": f"Score bajo ({best_score:.1f})"}

    return {
        "bet": best_bet,
        "covered_numbers": OUTSIDE_BETS[best_bet],
        "gale_level": 0,
        "reasoning": f"Mejor apuesta: {best_bet} (Score: {best_score:.1f})"
    }


# ==============================================================================
# LÓGICA DE PREDICCIÓN (MODOS)
# ==============================================================================

def find_best_number_bet() -> Dict[str, Any]:
    """Lógica de predicción para el modo NEIGHBORS (LA FUNCIÓN PESADA)"""
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
            "reasoning": "Fase de aprendizaje inicial, usando par por defecto.",
            "gale_level": 0
        }

    lookback_len = min(len(history), IA_CONFIG_NEIGHBORS["LOOKBACK"])
    history_window = history[-lookback_len:]
    
    distance_patterns = analyze_distance_patterns(history_window)
    is_stable = distance_patterns["is_stable"]
    
    if IA_CONFIG_NEIGHBORS["USE_STABILITY_FILTER"] and not is_stable:
        return {
            "base_numbers": [],
            "covered_numbers": [],
            "coverage_count": 0,
            "reasoning": f"PAUSA: Volatilidad alta detectada ({distance_patterns['volatility']:.1f} > {IA_CONFIG_NEIGHBORS['VOLATILITY_THRESHOLD']})",
            "gale_level": 0,
            "bet": "IDLE" # <-- Asegúrate de que las predicciones nulas tengan 'bet'
        }

    counts = Counter(history_window)
    total_spins = len(history_window)
    
    hot_zones = analyze_hot_zones(history_window)
    breakouts = detect_breakouts(history_window)
    clusters = detect_clusters(history_window)
    global sector_momentum
    sector_momentum = analyze_sector_momentum(history_window)
    
    analysis_data = {
        'frequency': {n: counts.get(n, 0) / total_spins for n in ROULETTE_ORDER},
        'recency': {n: (lookback_len - number_last_seen[n]) / lookback_len if number_last_seen[n] < lookback_len else 0.0 for n in ROULETTE_ORDER},
        'hot_zones': hot_zones,
        'breakouts': breakouts
    }
    
    dfreq, recency, hot_zones, breakouts = {}, {}, {}, {}
    for n in ROULETTE_ORDER:
        dfreq[n] = analysis_data['frequency'].get(n, 0)
        recency[n] = analysis_data['recency'].get(n, 0)
        hot_zones[n] = analysis_data['hot_zones'].get(n, 1.0)
        breakouts[n] = analysis_data['breakouts'].get(n, 0.0)

    neigh_sets = {n: set(get_neighbour_set(n, neighbours)) for n in ROULETTE_ORDER}
    
    def composite_score(n):
        f = dfreq[n] * 12.0
        r = recency[n] * 9.0
        h = (hot_zones[n] - 1.0) * 10.0 if hot_zones[n] > 1.0 else (hot_zones[n] - 1.0) * 5.0
        b = breakouts[n] * 6.0
        return f + r + h + b

    if IA_CONFIG_NEIGHBORS["DYNAMIC_TOP_NUMBERS"]:
        all_scores = [(n, composite_score(n)) for n in ROULETTE_ORDER]
        all_scores.sort(key=lambda x: x[1], reverse=True)
        dynamic_top_count = max(
            IA_CONFIG_NEIGHBORS["MIN_TOP_NUMBERS"],
            min(IA_CONFIG_NEIGHBORS["MAX_TOP_NUMBERS"], int(len(history) / 10))
        )
        top_numbers = [n for n, score in all_scores[:dynamic_top_count]]
    else:
        top_numbers = ROULETTE_ORDER

    pair_scores = {}
    for i in top_numbers:
        for j in top_numbers:
            if i >= j: continue
            
            set_i = neigh_sets[i]
            set_j = neigh_sets[j]
            overlap = len(set_i & set_j)
            
            if overlap > 0:
                continue

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
            if distance_patterns["trend"] > 1.5 and get_wheel_distance(i, j) > distance_patterns["avg_distance"]:
                pattern_score = get_weight("PATTERN_WEIGHT") * 5
            elif distance_patterns["trend"] < -1.5 and get_wheel_distance(i, j) < distance_patterns["avg_distance"]:
                pattern_score = get_weight("PATTERN_WEIGHT") * 5

            sector_bonus = 0
            if sector_momentum:
                idx1 = ROULETTE_ORDER.index(i); idx2 = ROULETTE_ORDER.index(j)
                sector1 = f"sector_{(idx1 // 9) + 1}"; sector2 = f"sector_{(idx2 // 9) + 1}"
                sector_bonus = (sector_momentum.get(sector1, 0) + sector_momentum.get(sector2, 0)) * 5
            
            p3 = 1.0 - ((1.0 - (C/37.0)) ** 3)
            accuracy_score = get_weight("ACCURACY_WEIGHT") * (p3 * 10)
            
            total_score = (
                coverage_score + coverage_quality_score + frequency_score + 
                recency_score + hot_zone_score + diversity_score + 
                performance_score + cluster_bonus + breakout_bonus + 
                pattern_score + sector_bonus + accuracy_score
            )
            
            pair_scores[(i, j)] = total_score
    
    if not pair_scores:
        chosen = None
    else:
        sorted_pairs = sorted(pair_scores.items(), key=lambda item: item[1], reverse=True)
        chosen = sorted_pairs[0][0]
        
    if chosen is None:
        # print("⚠️ FALLBACK: No se encontraron pares sin solapamiento en el Top-N. Buscando en todos los 37 números...")
        fallback_best_pair = None
        fallback_best_score = float('-inf')
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
            # print(f"✅ Fallback exitoso: {base_nums} (encontrado en búsqueda completa)")
        else:
            # print("🚨 ERROR CRÍTICO: No se encontró NINGÚN par sin solapamiento. Usando par de emergencia [0, 11].")
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
        if pair_scores: # Asegurarse de que pair_scores no esté vacío
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
                # print("🔄 CAMBIO ESTRATÉGICO DE PAR")

    covered = neigh_sets[base_nums[0]] | neigh_sets[base_nums[1]]
    C = len(covered)
    p3 = 1.0 - ((1.0 - (C/37.0)) ** 3)
    prediction_confidence = min(p3 * confidence_mult * get_pair_performance_score(tuple(sorted(base_nums))), 0.95)
    best_score = pair_scores.get(tuple(sorted(base_nums)), 0)

    return {
        "bet": f"Par {base_nums[0]},{base_nums[1]}", # <-- Añadido 'bet'
        "base_numbers": base_nums,
        "covered_numbers": sorted(list(covered)),
        "coverage_count": C,
        "gale_level": 0,
        "confidence": round(prediction_confidence, 2),
        "reasoning": f"Par {base_nums[0]},{base_nums[1]} | Fase: {phase} v3.8 - Score: {best_score:.1f}"
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
# --- ¡NUEVO! FUNCIÓN DE TAREA EN SEGUNDO PLANO ---
# ==============================================================================

def create_signal_in_background(number: int):
    """
    Esta es la función LENTA que se ejecuta en segundo plano (el "ayudante").
    Calcula la señal y actualiza el estado global.
    """
    global game_state, active_bet, latest_data_for_frontend, new_update_event, current_mode, stats

    logger.info(f"[BG Task] Iniciando cálculo de señal para el N°{number} (Modo: {current_mode})...")
    
    try:
        current_phase = get_current_phase() # Necesita recalcular la fase
        should_generate_signal = (
            (IA_CONFIG_NEIGHBORS["FORCE_ENTRY"] if current_mode == "neighbors" else IA_CONFIG_OUTSIDE["FORCE_ENTRY"]) 
            or (current_phase == "OPTIMAL")
        )
        
        if should_generate_signal:
            prediction = {}
            if current_mode == "neighbors":
                prediction = find_best_number_bet()
            else: # current_mode == "outside"
                prediction = find_best_outside_bet()

            if prediction and prediction.get("bet") not in ["IDLE", "LEARNING"] and prediction.get("covered_numbers"):
                active_bet = prediction
                active_bet["mode"] = current_mode # Sellar el modo de la apuesta
                game_state = "AWAITING_INITIAL_RESULT"
                stats[current_mode]["total_signals"] += 1
                
                bet_name = prediction['bet'] if current_mode == 'outside' else prediction.get('base_numbers', 'N/A')
                logger.info(f"🔔 [BG Task] ¡NUEVA SEÑAL GENERADA!: {bet_name}")
                
                # --- ¡NOTIFICAR AL FRONTEND! ---
                # Toca la campana para que el long-polling lo recoja.
                final_data = {
                    "action": "NEW_BET", # El frontend reaccionará a esto
                    "response": {"action": "NEW_BET", "bet_details": active_bet},
                    "last_number": number,
                    "history": history[-20:],
                    "stats": get_all_stats()
                }
                latest_data_for_frontend = final_data
                new_update_event.set()
                new_update_event.clear()
            else:
                reason = prediction.get("reasoning", "Sin señal válida")
                logger.info(f"[BG Task] Cálculo terminado. {reason}.")
                game_state = "IDLE" # Vuelve a estar disponible
        else:
             logger.info(f"[BG Task] Cálculo omitido (should_generate_signal=False).")
             game_state = "IDLE" # Vuelve a estar disponible

    except Exception as e:
        logger.error(f"¡¡¡ERROR CRÍTICO en Background Task!!!: {e}")
        game_state = "IDLE" # Resetea el estado para no bloquearse


# ==============================================================================
# ENDPOINTS DE FASTAPI
# ==============================================================================

# --- ¡NUEVO! Endpoint para carga masiva ---
@app.post("/bulk_add")
async def bulk_add_numbers(data: BulkNumbersInput):
    """
    Recibe una lista de números y los procesa rápidamente
    sin ejecutar la IA en cada paso, solo al final.
    """
    global history, latest_data_for_frontend, new_update_event, game_state
    
    # Procesa la mayoría de números sin lógica pesada
    # Asumimos que la mayoría son de aprendizaje
    for number in data.numbers:
        if not (0 <= number <= 36):
            continue
        
        history.append(number)
        if current_mode == "neighbors":
            update_number_tracking(number)
    
    logger.info(f"Carga Masiva: {len(data.numbers)} números añadidos. Historial total: {len(history)}.")
    
    # Ahora, forzamos UNA SOLA actualización de estado al final
    game_state = "IDLE" # Forza a que la próxima petición /process_number piense
    
    final_data = {
        "action": "BULK_UPDATE",
        "message": f"{len(data.numbers)} números añadidos.",
        "last_number": history[-1] if history else None,
        "history": history[-20:],
        "stats": get_all_stats()
    }
    
    latest_data_for_frontend = final_data
    new_update_event.set()
    new_update_event.clear()
    
    return {"status": "success", "message": f"{len(data.numbers)} números añadidos."}


# --- ¡MODIFICADO! Endpoint principal para procesar números ---
@app.post("/process_number")
async def process_number(data: NumberInput, tasks: BackgroundTasks):
    """
    Procesa un número (el "chef").
    Responde rápido y delega el trabajo pesado (cálculo de señal)
    a una Tarea en Segundo Plano (el "ayudante").
    """
    global game_state, active_bet, history, latest_data_for_frontend, cooldown_rounds, recent_pairs, last_pair_used, same_pair_streak, pair_performance, is_stable, zero_protection_active
    
    number = data.number
    
    if not (0 <= number <= 36):
        raise HTTPException(status_code=400, detail="Número inválido.")

    history.append(number)
    
    if current_mode == "neighbors":
        update_number_tracking(number)
    
    current_phase = get_current_phase()
    response: Dict[str, Any] = {"action": "WAIT"}
    
    # --- LÓGICA DE VICTORIA/DERROTA (RÁPIDA) ---
    bet_active = (game_state not in ["IDLE", "LEARNING", "COOLDOWN", "ANALYZING_BG"])
    
    if bet_active and active_bet:
        bet_mode = active_bet["mode"] # 'neighbors' o 'outside'
        mode_stats = stats[bet_mode]
        
        is_win = number in active_bet["covered_numbers"]
        
        if is_win:
            mode_stats["total_wins"] += 1
            mode_stats["current_win_streak"] += 1
            mode_stats["recent_results"].append(True)
            
            win_level = "INITIAL"
            if game_state == "AWAITING_INITIAL_RESULT":
                mode_stats["wins_initial"] += 1
                win_level = "INITIAL"
            elif game_state == "AWAITING_G1_RESULT":
                mode_stats["wins_g1"] += 1
                win_level = "G1"
            elif game_state == "AWAITING_G2_RESULT":
                mode_stats["wins_g2"] += 1
                win_level = "G2"

            if bet_mode == "neighbors" and "base_numbers" in active_bet:
                bi, bj = active_bet["base_numbers"]
                pair_key = tuple(sorted((bi, bj)))
                if pair_key not in pair_performance:
                    pair_performance[pair_key] = {'wins': 0, 'losses': 0, 'uses': 0, 'recent_wins': 0, 'recent_uses': 0}
                pair_performance[pair_key]['wins'] += 1
                pair_performance[pair_key]['uses'] += 1
                pair_performance[pair_key]['recent_wins'] = min(pair_performance[pair_key].get('recent_wins', 0) + 1, 5)
                pair_performance[pair_key]['recent_uses'] = min(pair_performance[pair_key].get('recent_uses', 0) + 1, 5)

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

            game_state = "IDLE" # ¡LISTO PARA OTRO PEDIDO!
            active_bet = None
            cooldown_rounds = IA_CONFIG_NEIGHBORS["COOLDOWN_ROUNDS"] if bet_mode == "neighbors" else IA_CONFIG_OUTSIDE["COOLDOWN_ROUNDS"]
            logger.info(f"✅ [{bet_mode.upper()}] VICTORIA! Nro: {number} | Nivel: {win_level} | Racha: {mode_stats['current_win_streak']}")

        else: # No es victoria (Gale)
            mode_stats["recent_results"].append(False)
            
            if game_state == "AWAITING_INITIAL_RESULT":
                game_state = "AWAITING_G1_RESULT"
                response = {"action": "G1", "bet_details": active_bet}
            
            elif game_state == "AWAITING_G1_RESULT":
                game_state = "AWAITING_G2_RESULT"
                response = {"action": "G2", "bet_details": active_bet}
            
            else: # Pérdida final (G2)
                mode_stats["total_losses"] += 1
                mode_stats["current_win_streak"] = 0

                if bet_mode == "neighbors" and "base_numbers" in active_bet:
                    bi, bj = active_bet["base_numbers"]
                    pair_key = tuple(sorted((bi, bj)))
                    if pair_key in pair_performance:
                        pair_performance[pair_key]['losses'] = pair_performance[pair_key].get('losses', 0) + 1
                        pair_performance[pair_key]['uses'] = pair_performance[pair_key].get('uses', 0) + 1
                        pair_performance[pair_key]['recent_uses'] = min(pair_performance[pair_key].get('recent_uses', 0) + 1, 5)
                        pair_performance[pair_key]['recent_wins'] = 0
                    
                    recent_pairs.append(pair_key)
                    last_pair_used = pair_key
                
                response = {"action": "LOSS", "correct_number": number, "bet_details": active_bet}
                game_state = "IDLE" # ¡LISTO PARA OTRO PEDIDO!
                active_bet = None
                cooldown_rounds = IA_CONFIG_NEIGHBORS["COOLDOWN_ROUNDS"] if bet_mode == "neighbors" else IA_CONFIG_OUTSIDE["COOLDOWN_ROUNDS"]
                logger.info(f"❌ [{bet_mode.upper()}] PÉRDIDA. Nro: {number} | Racha reiniciada.")

    # --- LÓGICA DE COOLDOWN (RÁPIDA) ---
    elif cooldown_rounds > 0:
        cooldown_rounds -= 1
        response = (
            {"action": "READY", "message": "Cooldown terminado"} if cooldown_rounds == 0
            else {"action": "COOLDOWN", "rounds_left": cooldown_rounds}
        )
        if cooldown_rounds == 0:
            game_state = "IDLE" # ¡LISTO!
    
    # --- ¡LÓGICA DE NUEVA SEÑAL (MODIFICADA!) ---
    elif game_state in ["IDLE", "LEARNING"] and current_phase != "LEARNING":
        
        # ¡AQUÍ ESTÁ LA MAGIA!
        
        # 1. Cambia el estado para que no se pida otro pastel
        #    mientras el ayudante está ocupado.
        game_state = "ANALYZING_BG" # Un nuevo estado: "pensando en 2do plano"
        
        # 2. Responde "ESPERA" inmediatamente (esto arregla el Error de Red)
        response = {"action": "WAIT", "message": "Analizando..."}
        
        # 3. Programa la tarea pesada para DESPUÉS de responder.
        tasks.add_task(create_signal_in_background, number)
        
        logger.info(f"Petición {number} recibida. Respondiendo 'WAIT' y enviando cálculo a 2do plano.")
    
    # --- LÓGICA DE APRENDIZAJE (RÁPIDA) ---
    else: # Fase de aprendizaje o estado ANALYZING_BG (ya ocupado)
        msg = f"{len(history)}/{IA_CONFIG_NEIGHBORS['LEARNING_PHASE_MIN']}"
        if game_state == "ANALYZING_BG":
            msg = "Analizando señal anterior..."
        
        response = {"action": "LEARNING", "message": msg}

    # --- Empaquetar datos para el frontend ---
    # (Envía la respuesta RÁPIDA: WIN, LOSS, GALE, COOLDOWN, o WAIT)
    final_data = {
        "action": response.get("action", "UPDATE"),
        "response": response,
        "last_number": number,
        "history": history[-20:],
        "stats": get_all_stats()
    }
    
    # Notificar al long-polling
    global latest_data_for_frontend, new_update_event
    latest_data_for_frontend = final_data
    new_update_event.set()
    new_update_event.clear()
    
    return final_data


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
    
    logger.info(f"🔄 MODO CAMBIADO A: {current_mode.upper()}")
    
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
        # Espera a que la "campana" (new_update_event) suene
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
        logger.error(f"Error en initial_state: {e}")
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

    game_state = "IDLE"
    active_bet = None
    cooldown_rounds = 0
    
    deleted_number = history.pop()
    
    if current_mode == 'neighbors':
        temp_history = list(history)
        number_last_seen = {n: 0 for n in ROULETTE_ORDER}
        for i, num in enumerate(reversed(temp_history)):
            if num in number_last_seen and number_last_seen[num] == 0:
                if i > 0:
                    number_last_seen[num] = i 
        
        max_seen = len(temp_history)
        for n in ROULETTE_ORDER:
            if n not in temp_history:
                number_last_seen[n] = max_seen
    
    logger.info(f"⏪ NÚMERO BORRADO: {deleted_number}. Datos recalculados.")

    response = {
        "action": "DELETE_LAST",
        "message": f"Número {deleted_number} borrado.",
        "stats": get_all_stats(),
        "history": history[-20:],
        "last_number": history[-1] if history else None
    }
    
    global latest_data_for_frontend, new_update_event
    latest_data_for_frontend = response
    new_update_event.set()
    new_update_event.clear()

    return {"status": "success", "stats": get_all_stats()}


def reset_stats_internal():
    """Función interna para resetear estadísticas."""
    global stats
    for mode in ["neighbors", "outside"]:
        stats[mode] = {
            "total_wins": 0, "total_losses": 0,
            "wins_initial": 0, "wins_g1": 0, "wins_g2": 0,
            "total_signals": 0, "current_win_streak": 0,
            "recent_results": deque(maxlen=IA_CONFIG_NEIGHBORS["ADAPTATION_WINDOW"] if mode == "neighbors" else 20)
        }

@app.post("/reset_stats")
def reset_stats():
    """Resetea solo las estadísticas, mantiene el historial."""
    reset_stats_internal()
    logger.info("🔄 RESET DE ESTADÍSTICAS - El historial se mantiene.")
    return {"status": "success", "stats": get_all_stats()}


def reset_all_internal():
    """Función interna para reseteo completo."""
    global history, game_state, active_bet, cooldown_rounds, number_last_seen, pattern_memory, cluster_cache, recent_pairs
    
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
    logger.info("🔄 RESET COMPLETO - Sistema reiniciado a fase de aprendizaje")
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
        return {
            "mode": "neighbors",
            "phase": get_current_phase(),
            "confidence": get_confidence_multiplier(),
            "is_stable": is_stable,
            "distance_patterns": analyze_distance_patterns(history[-IA_CONFIG_NEIGHBORS["STABILITY_WINDOW"]:]),
            "current_recommendation": reco,
            "history_length": len(history),
            "game_state": game_state,
            "sector_momentum": sector_momentum,
            "config": IA_CONFIG_NEIGHBORS
        }
    else: # outside
        reco = find_best_outside_bet()
        trends = analyze_trends(history)
        return {
            "mode": "outside",
            "trends": trends,
            "streaks": {
                "color": analyze_streaks(history, "color"),
                "parity": analyze_streaks(history, "parity"),
                "range": analyze_streaks(history, "range"),
            },
            "current_recommendation": reco,
            "history_length": len(history),
            "game_state": game_state,
            "zero_protection": zero_protection_active,
            "config": IA_CONFIG_OUTSIDE
        }

# ==============================================================================
# BLOQUE DE INICIO DEL SERVIDOR (MODIFICADO PARA RAILWAY)
# ==============================================================================

if __name__ == "__main__":
    
    # --- ¡CORREGIDO! Obtiene el puerto de Railway o usa 8000 por defecto
    port = int(os.environ.get("PORT", 8000))
    
    # Mantenemos tus logs de inicio, pero usando la variable 'port'
    print("=" * 70)
    print("🚀 SERVIDOR IA COMBINADO v3.8 (Optimizado para Acertividad)")
    print("=" * 70)
    
    # --- ¡CORREGIDO! Muestra 0.0.0.0 y el puerto dinámico
    print(f"   ✓ ¡NUEVO! Sirviendo HTML en http://0.0.0.0:{port}/")
    
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
    print("     ✓ Análisis de Rachas y Cortes")
    print("     ✓ Estrategia Momentum pura (sin contradicciones)")
    print("=" * 70)
    print("   ✓ Estadísticas GALE separadas por modo.")
    print("   ✓ Cooldown unificado: 5 rondas.")
    print("=" * 70)

    # --- ¡LA LÍNEA MÁS IMPORTANTE! ---
    # 1. host="0.0.0.0" (para aceptar conexiones externas)
    # 2. port=port (para usar el puerto dinámico de Railway)
    uvicorn.run(app, host="0.0.0.0", port=port)
