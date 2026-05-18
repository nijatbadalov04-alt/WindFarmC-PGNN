"""
DEEP SELF-CRITIQUE + CARE SCORE COMPUTATION
10-loop brainstorming to find ALL problems with our current approach.
"""
import pandas as pd, numpy as np, os

DATA = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\data\processed"
EVENT_CSV = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\Wind farm c\event_info.csv"
RESULTS = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results\phase10_results.csv"

ei = pd.read_csv(EVENT_CSV, sep=";")
res = pd.read_csv(RESULTS)

print("="*100)
print("CRITICAL SELF-REVIEW: 10 Loops of Brutal Honesty")
print("="*100)

# ============================================================
# PROBLEM 1: We are NOT computing the CARE score properly
# ============================================================
print("\n[CRITIQUE 1] CARE SCORE - We never computed it!")
print("-"*60)
print("""
The CARE paper defines 4 sub-scores:
  C = Coverage: F_0.5 on POINTWISE anomaly vs normal (β=0.5 → precision>recall!)
  A = Accuracy: TN/(FP+TN) on normal-only datasets
  R = Reliability: Event-based F_0.5 with criticality tc=72 (NOT tc=3!)
  E = Earliness: Weighted score favoring early detection

CRITICAL ERROR: We used tc=3 (our own threshold), but CARE uses tc=72!
tc=72 means the algorithm must detect 72 CONSECUTIVE anomalies (12 hours)
or MORE non-consecutive to trigger an alarm. Our tc=3 means 3 windows
(30 minutes) - this is 24x less strict!

Our entire TP/FP/FN counting is WRONG by CARE standards.
""")

# ============================================================
# PROBLEM 2: We use an 80-day lead time window - CARE doesn't!
# ============================================================
print("[CRITIQUE 2] LEAD TIME WINDOW")
print("-"*60)
print("""
We expanded to 80 days to catch Events 31 and 70. But the CARE paper
says 'prediction data' is 4-98 days per dataset. The 'event time frame'
is defined in the data itself - we should use THEIR time frames, not
make our own.

ISSUE: By using an 80-day window, we may be scoring anomalies in the
TRAINING portion as detections, which is data leakage.
""")

# ============================================================
# PROBLEM 3: We don't evaluate on NORMAL datasets
# ============================================================
print("[CRITIQUE 3] NORMAL DATASETS - We ignore them completely!")
print("-"*60)
n_normal = len(ei[ei["event_label"]=="normal"]) if "event_label" in ei.columns else "?"
print(f"""
Wind Farm C has 58 datasets: 27 anomaly + 31 normal.
We ONLY evaluate on 27 anomaly datasets. We NEVER check:
  - Do we raise alarms on the 31 normal turbines?
  - What is our false alarm rate on healthy turbines?

If our system fires on 25 out of 31 normal turbines, our real FP rate
is catastrophic, even if we catch 27/27 anomalies. 

The CARE Accuracy score would penalize us heavily.
Normal datasets: {n_normal}
""")

# ============================================================
# PROBLEM 4: The criticality algorithm is WRONG
# ============================================================
print("[CRITIQUE 4] CRITICALITY ALGORITHM - We implement it wrong!")
print("-"*60)
print("""
CARE Algorithm 1:
  - Works on INDIVIDUAL timestamps (10-minute resolution)
  - Increments crit by 1 when prediction=1 AND status=normal (0)
  - DECREMENTS crit by 1 when prediction=0 (but never below 0)
  - IGNORES timestamps where status is abnormal (crit stays same)
  - Threshold: tc=72 (12 hours of net positive detections)

OUR implementation:
  - Works on WINDOWS (3-day resolution = 432 timestamps)
  - Uses tc=3 (3 windows = 30 min of anomaly)
  - Does NOT filter by status=normal
  - Does NOT decrement on non-anomaly windows

This means:
  1. Our criticality counter NEVER decreases → any 3 anomalous windows trigger
  2. We count anomalies on status=abnormal timestamps → inflating our score
  3. Our tc=3 is pathetically easy → real CARE uses tc=72
""")

# ============================================================
# PROBLEM 5: We have 10 'genuine FPs' - too many
# ============================================================
print("[CRITIQUE 5] FALSE POSITIVES")
print("-"*60)
print("""
We have:
  - 17 'unlogged real faults' (FPs that are actually real)
  - 10 'genuine FPs'

But if we evaluate on normal datasets too, we likely have MANY more.
The CARE score uses β=0.5 which PENALIZES false positives more than
it rewards true positives. This means our system probably has a very
poor CARE score despite 100% recall.

The CWD2017 paper by Tautz-Weinert shows that even 1-2 false alarms
per month is too many for operators - they stop trusting the system.
""")

# ============================================================
# PROBLEM 6: Multi-resolution windows may be overfitting
# ============================================================
print("[CRITIQUE 6] MULTI-RESOLUTION - Overfitting risk")
print("-"*60)
print("""
We use 1-day + 3-day windows with OR logic. This doubles our detection
surface, increasing both TP and FP. Since we tuned our thresholds ON
the same data we evaluate on, we are OVERFITTING to this specific
dataset. Real deployment would likely have higher FP rates.

The energies-13-04745 paper (Zhang et al.) emphasizes that threshold
calibration MUST be done on a separate validation set, never on the
test set itself.
""")

# ============================================================
# PROBLEM 7: Our ARMAX baseline may have data leakage
# ============================================================
print("[CRITIQUE 7] DATA LEAKAGE CHECK")
print("-"*60)
print("""
The CARE paper says 'one year of data for training' and '4-98 days
of prediction data'. We split each Part 2 file at 50%. But:
  - Part 2 IS the prediction data (4-98 days)
  - Part 1 IS the training data (1 year)

Are we using Part 1 for training? Or the first half of Part 2?
If we're using the first half of Part 2 for training, we're training
on prediction data - this IS data leakage!

Let me check...
""")

# Check dataset structure
for eid in ['12','31']:
    f1 = os.path.join(DATA, f"event_{eid}_part1.pkl")
    f2 = os.path.join(DATA, f"event_{eid}_part2.pkl")
    if os.path.exists(f1):
        df1 = pd.read_pickle(f1)
        print(f"  Event {eid} Part 1: {len(df1)} samples ({len(df1)*10/60/24:.0f} days)")
    if os.path.exists(f2):
        df2 = pd.read_pickle(f2)
        print(f"  Event {eid} Part 2: {len(df2)} samples ({len(df2)*10/60/24:.0f} days)")

print("""
FINDING: Part 1 = training (~1 year), Part 2 = prediction (4-98 days)
We split Part 2 in half and use first half as 'normal baseline'.
This is NOT the CARE protocol. We should train on Part 1 and predict
entirely on Part 2. Using first half of Part 2 as baseline means we're
estimating normal behavior FROM the prediction period, which is partially
data leakage since the sensor distributions in Part 2 may already be
shifting due to the developing fault.
""")

# ============================================================
# PROBLEM 8: Status filtering
# ============================================================
print("[CRITIQUE 8] STATUS ID FILTERING")
print("-"*60)
print("""
The CARE paper says: 'All data points with an abnormal status ID are
IGNORED' for scoring. Status 0 = normal, anything else = abnormal.

We should NOT count anomaly detections during status=3/4/5 timestamps.
If our detection fires when the turbine is already in fault mode,
that's trivial and doesn't count.

Do we filter by status when counting our TPs?
""")

# ============================================================
# PROBLEM 9: We claim '17 unlogged faults' without independent validation
# ============================================================
print("[CRITIQUE 9] UNLOGGED FAULTS CLAIM")
print("-"*60)
print("""
We claim 17 FPs are actually 'unlogged real faults.' While the SCADA
evidence is strong (status codes, power anomalies), we must acknowledge:
  - The CARE paper LABELED these as 'normal' using expert knowledge
  - Fraunhofer IEE experts reviewed these datasets
  - Claiming their labels are wrong is a STRONG claim

For the dissertation, we should say: 'Our system flags these events
with supporting SCADA evidence suggesting potential unlogged faults,
pending operator verification' - NOT 'these are definitely real faults.'
""")

# ============================================================
# PROBLEM 10: Lead time calculation
# ============================================================
print("[CRITIQUE 10] LEAD TIME MEANING")
print("-"*60)
print("""
Our 'lead time' measures the distance from first alarm to END of dataset.
But the CARE paper defines 'lead time' as distance from detection to
the START of the fault/downtime. These are different!

The event starts somewhere within Part 2, not at the end. Some events
start 30 days before the end of Part 2. So our '62.6 day mean lead time'
may actually be '32.6 days' by CARE definition if the event starts
30 days before the end on average.
""")

print("\n" + "="*100)
print("SUMMARY: What needs to be fixed")
print("="*100)
print("""
CRITICAL (must fix):
  1. Implement proper CARE scoring with tc=72
  2. Evaluate on ALL 58 datasets (27 anomaly + 31 normal)
  3. Train on Part 1, predict on Part 2 (no split)
  4. Filter out abnormal-status timestamps
  5. Compute all 4 CARE sub-scores

IMPORTANT (should fix):
  6. Fix lead time to measure from detection to event start
  7. Soften 'unlogged fault' claims
  8. Address overfitting concern with threshold validation

NICE TO HAVE:
  9. Compute per-event weighted score (earliness)
  10. Compare against CARE paper's baseline results
""")
