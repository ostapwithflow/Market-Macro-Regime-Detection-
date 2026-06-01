import sys
import logging
import numpy as np
import pandas as pd
from datetime import timedelta

# Reconfigure stdout/stderr to UTF-8 to support Cyrillic output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Suppress logs for neat output
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data.fred_client import FredClient
from data.macro_features import build_emission_vector
from models.ood_gatekeeper import OODGatekeeper
from models.hmm_core import RegimeEngineHMM
from data.economic_calendar import EconomicCalendar


def run_macro_regime_assessment():
    """
    Macro-Only Regime Assessment (Slow Tower / Layer 1).
    Classifies the current structural market regime as CALM / TRANSITION / STRESS
    using federal liquidity data and yield curve signals from FRED.
    """
    print("\n[INFO] Ініціалізація Macro Regime Engine (Slow Tower)...")
    
    start_date = "2010-01-01"
    
    # 1. Завантаження макро-даних FRED
    fred_client = FredClient()
    raw_macro = fred_client.fetch_all_macro_data(start_date=start_date)
    
    if raw_macro.empty:
        print("[ERROR] Не вдалося завантажити макро-дані з FRED. Перевірте FRED_API_KEY.")
        return
    
    X_hmm = build_emission_vector(raw_macro)
    
    # 2. Календар подій
    cal = EconomicCalendar()
    cal.load_all_events()
    
    # 3. Навчання моделей на останньому вікні (5 років)
    window_size = 252 * 5
    if len(X_hmm) < window_size:
        window_size = len(X_hmm)
        
    X_hmm_train = X_hmm.tail(window_size)
    
    # OOD Gatekeeper (MCD)
    gatekeeper = OODGatekeeper()
    gatekeeper.fit(X_hmm_train)
    ood_mask_train = gatekeeper.transform_predict(X_hmm_train)
    
    # HMM (3 states, 7 random restarts)
    hmm = RegimeEngineHMM(n_components=3, n_restarts=7, random_state=42)
    hmm.fit(X_hmm_train, ood_mask_train)
    
    # 4. Оцінка поточного дня (t) та прогноз на завтра (t+1)
    current_date = X_hmm_train.index[-1]
    
    # HMM постеріорні ймовірності для сьогодні
    ood_mask_today = gatekeeper.transform_predict(X_hmm_train)
    regimes_df = hmm.predict_regimes(X_hmm_train, ood_mask_today)
    today_probs = regimes_df.iloc[-1][['Prob_CALM', 'Prob_TRANSITION', 'Prob_STRESS']].values.astype(float)
    is_today_ood = bool(regimes_df.iloc[-1]['is_OOD'])
    
    # Визначення завтрашнього дня
    tomorrow = current_date + timedelta(days=1)
    is_event_tomorrow = tomorrow in cal.all_event_dates
    
    # Predict tomorrow's macro probabilities from HMM transition matrix
    tomorrow_probs = np.dot(hmm.best_model.transmat_.T, today_probs)
    
    # 5. Розрахунок метрик
    regime_map = {0: "CALM", 1: "TRANSITION", 2: "STRESS"}
    winning_state = np.argmax(tomorrow_probs)
    regime_name = regime_map[winning_state]
    
    # Confidence: постеріорна впевненість
    confidence = tomorrow_probs[winning_state]
    # Stability: самоперехід HMM
    stability = hmm.best_model.transmat_[winning_state, winning_state]
    # Model Certainty
    certainty = 0.55 * confidence + 0.45 * stability
    
    # 6. Виведення фінального ASCII Дашборду
    print("\n" + "="*80)
    print("              MACRO REGIME ENGINE — SLOW TOWER (Layer 1)")
    print("="*80)
    print(f" Останнє оновлення:  {current_date.strftime('%Y-%m-%d')} (t)")
    print(f" Прогноз на сесію:   {tomorrow.strftime('%Y-%m-%d')} (t+1)")
    print(f" Аномалія (OOD):     {'[ALERT] OUT_OF_DISTRIBUTION' if is_today_ood else '[OK] NOMINAL (Normal)'}")
    print(f" Макро-Подія:        {'[YES] FOMC/CPI/NFP' if is_event_tomorrow else 'Ні'}")
    print("-" * 80)
    
    # Визначення кольору структурного режиму
    if regime_name == "CALM":
        reg_color = "\033[92m"  # Green
    elif regime_name == "TRANSITION":
        reg_color = "\033[93m"  # Yellow
    else:
        reg_color = "\033[91m"  # Red
        
    print(f" [MACRO REGIME]  Структурний режим:     {reg_color}{regime_name}\033[0m")
    print(f"                 Ймовірності станів:    CALM: {tomorrow_probs[0]*100:.1f}% | TRANSITION: {tomorrow_probs[1]*100:.1f}% | STRESS: {tomorrow_probs[2]*100:.1f}%")
    print(f"                 Стабільність режиму:   {stability*100:.1f}%")
    print(f"                 Впевненість (Cert):    {certainty*100:.1f}%")
    print("-" * 80)
    
    # Transition matrix summary
    print(" [TRANSITION MATRIX]")
    labels = ["CALM", "TRANS", "STRESS"]
    header = "                 " + "  ".join(f"{l:>8s}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row_vals = "  ".join(f"{hmm.best_model.transmat_[i, j]*100:>7.1f}%" for j in range(3))
        print(f"   {label:>8s}  →  {row_vals}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_macro_regime_assessment()
