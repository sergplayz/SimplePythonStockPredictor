import yfinance as yf
import pandas as pd
import talib
import os
import csv
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

def calc_f(df):
    df_c = df.copy()
    df_c['DR'] = df_c['Close'].pct_change()
    df_c['SMA_5'] = df_c['Close'].rolling(window=5).mean()
    df_c['SMA_10'] = df_c['Close'].rolling(window=10).mean()
    df_c['SMA_20'] = df_c['Close'].rolling(window=20).mean()
    df_c['RSI_14'] = talib.RSI(df_c['Close'], timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df_c['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df_c['MACD'] = macd
    df_c['MACD_Signal'] = macdsignal
    df_c['MACD_Hist'] = macdhist
    df_c['Dir'] = (df_c['Close'].shift(-1) > df_c['Open'].shift(-1)).astype(int)
    df_c.dropna(inplace=True)
    return df_c

def init_log(lf='log.txt'):
    h = ['ts', 'sd', 'pv', 'pd', 'ad', 'acc']
    with open(lf, 'w', newline='') as f: csv.writer(f).writerow(h)
    print(f"Log initialized: '{lf}'.")

def log_s(lf='log.txt', sd=None, pv=None, pd=None, ad=None, acc=None):
    ts = datetime.now().isoformat()
    dr = [ts, sd, pv, pd, ad, acc]
    with open(lf, 'a', newline='') as f: csv.writer(f).writerow(dr)

def load_ls(lf='log.txt', ic=50, mts=100, ad=None):
    if not os.path.exists(lf) or os.stat(lf).st_size == 0:
        print(f"Log '{lf}' empty. Starting fresh.")
        return {'sdi': mts, 'pv': ic, 'acc': 0.0, 'ra': [], 'rp': []}
    try:
        with open(lf, 'r') as f:
            rdr = csv.DictReader(f)
            rws = list(rdr)
            if not rws: return {'sdi': mts, 'pv': ic, 'acc': 0.0, 'ra': [], 'rp': []}
            lr = rws[-1]
            lsds = lr['sd']
            lpv = float(lr['pv'])
            lma = float(lr['acc'])
            ra_l = [float(rw['ad']) for rw in rws[1:]]
            rp_l = [float(rw['pd']) for rw in rws[1:]]
            pd_d = pd.to_datetime(lsds, errors='coerce')
            if pd.isna(pd_d): return {'sdi': mts, 'pv': ic, 'acc': 0.0, 'ra': [], 'rp': []}
            if ad is None: return {'sdi': mts, 'pv': ic, 'acc': 0.0, 'ra': [], 'rp': []}
            if ad.tz is not None: lsd_dt = pd_d.tz_localize(ad.tz)
            else: lsd_dt = pd_d
            try:
                sdi = ad.get_loc(lsd_dt) + 1
            except KeyError:
                nvd = ad[ad > lsd_dt]
                if not nvd.empty: sdi = ad.get_loc(nvd.index[0])
                else: return {'sdi': mts, 'pv': ic, 'acc': 0.0, 'ra': [], 'rp': []}
            print(f"Resuming from {lsds} with PV ${lpv:.2f}.")
            return {'sdi': sdi, 'pv': lpv, 'acc': lma, 'ra': ra_l, 'rp': rp_l}
    except Exception as e:
        print(f"Error loading log: {e}. Starting fresh.")
        return {'sdi': mts, 'pv': ic, 'acc': 0.0, 'ra': [], 'rp': []}

t_sym = 'NVDA'
ic_start = 50
mts = 100
pi_min = 5

init_log()
nvda_d_raw = yf.Ticker(t_sym)
full_h_raw_df = nvda_d_raw.history(period='5y', interval='1d')
h_df_proc = calc_f(full_h_raw_df)
all_d_daily = h_df_proc.index

li = load_ls(lf='log.txt', ic=ic_start, mts=mts, ad=all_d_daily)

si_h_loop = li['sdi']
rt_cap = li['pv']
rt_acts = li['ra']
rt_preds = li['rp']

rt_pvs = [float(ic_start)]
if len(rt_acts) > 0:
    with open('log.txt', 'r') as f:
        rdr = csv.DictReader(f)
        for r in rdr: rt_pvs.append(float(r['pv']))

scaler = StandardScaler()
model = LogisticRegression(random_state=42, solver='liblinear')

cs_idx = si_h_loop

while True:
    cd_sim = all_d_daily[cs_idx]
    today = pd.Timestamp.now(tz='America/New_York').normalize()

    if cd_sim < today:
        nd_sim = all_d_daily[cs_idx + 1] if cs_idx + 1 < len(all_d_daily) else None
        if nd_sim is None or cs_idx + 1 >= len(all_d_daily) - 1:
            print(f"End of historical data: {cd_sim.strftime('%Y-%m-%d')}. Fetching live.")
            cs_idx = len(all_d_daily) - 1
            continue

        ctd = h_df_proc.loc[all_d_daily.min():cd_sim].copy()
        X_rt = ctd.drop('Dir', axis=1)
        y_rt = ctd['Dir']

        if X_rt.empty or y_rt.empty or len(X_rt) < mts:
            print(f"Skipping {cd_sim.strftime('%Y-%m-%d')} (insufficient train data).")
            cs_idx += 1
            continue

        X_s_rt = scaler.fit_transform(X_rt)
        model.fit(X_s_rt, y_rt)

        X_nd_f = h_df_proc.loc[[nd_sim]].drop('Dir', axis=1)
        if X_nd_f.empty:
            print(f"Skipping pred for {nd_sim.strftime('%Y-%m-%d')} (empty features).")
            cs_idx += 1
            continue

        X_nd_s = scaler.transform(X_nd_f)
        pred = model.predict(X_nd_s)[0]
        act_dir = h_df_proc.loc[nd_sim]['Dir']

        nd_open = full_h_raw_df.loc[nd_sim]['Open']
        nd_close = full_h_raw_df.loc[nd_sim]['Close']

    else:
        print(f"\n--- Fetching Live Data for {today.strftime('%Y-%m-%d')} ---")
        live_d_raw = nvda_d_raw.history(period='1d', interval=f'{pi_min}m')
        
        if live_d_raw.empty:
            print(f"No live data for {today.strftime('%Y-%m-%d')}. Waiting {pi_min} mins...")
            time.sleep(pi_min * 60)
            continue

        upd_full_h_raw_df = pd.concat([full_h_raw_df, live_d_raw]).drop_duplicates(keep='last')
        upd_h_df_proc = calc_f(upd_full_h_raw_df)
        
        if upd_h_df_proc.empty or len(upd_h_df_proc) < mts + 1:
            print(f"Not enough intraday data for live FE/pred. Waiting {pi_min} mins...")
            time.sleep(pi_min * 60)
            continue

        full_h_raw_df = upd_full_h_raw_df
        h_df_proc = upd_h_df_proc
        all_d_daily = h_df_proc.index
        cs_idx = len(all_d_daily) - 2
        cd_sim = all_d_daily[cs_idx]
        nd_sim = all_d_daily[cs_idx + 1]

        ctd = h_df_proc.loc[all_d_daily.min():cd_sim].copy()
        X_nd_f = h_df_proc.loc[[nd_sim]].drop('Dir', axis=1)
        act_dir = h_df_proc.loc[nd_sim]['Dir']

        nd_open = full_h_raw_df.loc[nd_sim]['Open']
        nd_close = full_h_raw_df.loc[nd_sim]['Close']

    X_rt = ctd.drop('Dir', axis=1)
    y_rt = ctd['Dir']

    if X_rt.empty or y_rt.empty or len(X_rt) < mts:
        print(f"Skipping {cd_sim.strftime('%Y-%m-%d')} (insufficient train data). Waiting {pi_min} mins...")
        time.sleep(pi_min * 60)
        continue

    X_s_rt = scaler.fit_transform(X_rt)
    model.fit(X_s_rt, y_rt)

    if X_nd_f.empty:
        print(f"Skipping pred for {nd_sim.strftime('%Y-%m-%d')} (empty features). Waiting {pi_min} mins...")
        time.sleep(pi_min * 60)
        continue

    X_nd_s = scaler.transform(X_nd_f)
    pred = model.predict(X_nd_s)[0]

    rt_preds.append(pred)
    rt_acts.append(act_dir)

    pl = (nd_close - nd_open) if pred == 1 else (nd_open - nd_close)
    drft = pl / nd_open
    rt_cap *= (1 + drft)
    rt_pvs.append(rt_cap)

    curr_acc = accuracy_score(rt_acts, rt_preds) if len(rt_acts) > 0 else 0.0
    log_s(sd=nd_sim.strftime('%Y-%m-%d %H:%M:%S'), pv=rt_cap, pd=pred, ad=act_dir, acc=curr_acc)

    print(f"Sim Date: {nd_sim.strftime('%Y-%m-%d %H:%M:%S')} | PV: ${rt_cap:.2f} | Pred: {pred} | Act: {act_dir} | Acc (Cum): {curr_acc:.4f}")

    cs_idx += 1

    if cd_sim >= today:
        print(f"Waiting {pi_min} mins for next live data point...")
        time.sleep(pi_min * 60)

    if cs_idx >= len(all_d_daily) - 1 and cd_sim < today:
        print(f"End of historical data: {cd_sim.strftime('%Y-%m-%d')}. Fetching live.")
        cs_idx = len(all_d_daily) - 1

