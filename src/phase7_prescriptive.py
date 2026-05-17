import pandas as pd
import os

RESULTS_DIR = r"C:\Users\nijat\OneDrive\Documents\WIND FIN\WindFarmC-Project\results"
res_path = os.path.join(RESULTS_DIR, "test_set_results.csv")

def main():
    if not os.path.exists(res_path):
        print("Run Phase 4 first.")
        return
        
    df = pd.read_csv(res_path)
    tp_df = df[df["type"] == "TP"]
    
    unique_events = tp_df["event_id"].unique()
    
    # Financial Model (Estimates in GBP £)
    COST_CATASTROPHIC = 150000  # Emergency repair, high downtime, full component replacement
    COST_PREDICTIVE = 45000     # Planned repair, minor part replacement
    
    # Prescriptive Variables
    CRANE_DISCOUNT = 10000      # Savings from flexible scheduling / off-peak crane booking
    CURTAILMENT_COST_PER_DAY = 50 # Lost power revenue from digitally derating turbine by 20%
    LIFE_EXTENSION_DAYS = 15
    
    total_baseline_cost = len(unique_events) * COST_CATASTROPHIC
    total_predictive_cost = 0
    total_prescriptive_cost = 0
    
    print("\n=== PHASE 7: PRESCRIPTIVE MAINTENANCE FINANCIAL SIMULATION ===\n")
    print(f"Total True Positive Detections Simulated: {len(unique_events)}")
    print(f"Catastrophic Failure Baseline Cost: £{total_baseline_cost:,}\n")
    
    for eid in unique_events:
        lead_time = tp_df[tp_df["event_id"] == eid]["lead_time_days"].max()
        
        pred_cost = COST_PREDICTIVE
        total_predictive_cost += pred_cost
        
        if lead_time < 15.0:
            presc_cost = COST_PREDICTIVE - CRANE_DISCOUNT + (CURTAILMENT_COST_PER_DAY * LIFE_EXTENSION_DAYS)
            action = f"DERATE 20% to extend life by {LIFE_EXTENSION_DAYS} days for cheap crane."
        else:
            presc_cost = COST_PREDICTIVE - CRANE_DISCOUNT
            action = "SCHEDULE cheap crane normally (Lead time sufficient)."
            
        total_prescriptive_cost += presc_cost
        
        print(f"Event {eid:>3} | Max Lead Time: {lead_time:>4.1f} days")
        print(f"  -> Predictive Cost:   £{pred_cost:,}")
        print(f"  -> Prescriptive Plan: {action}")
        print(f"  -> Prescriptive Cost: £{presc_cost:,}\n")
        
    print("=== FINANCIAL SUMMARY ===")
    print(f"Cost without AI (Catastrophic):  £{total_baseline_cost:,}")
    print(f"Cost with Predictive AI:         £{total_predictive_cost:,} (Saved: £{total_baseline_cost - total_predictive_cost:,})")
    print(f"Cost with Prescriptive AI:       £{total_prescriptive_cost:,} (Saved: £{total_baseline_cost - total_prescriptive_cost:,})")
    print(f"\nAdded Value of Industry-Lead Prescriptive Engine over standard AI: £{total_predictive_cost - total_prescriptive_cost:,}")

if __name__ == "__main__":
    main()
