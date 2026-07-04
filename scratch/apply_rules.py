import json

def apply_overrides():
    with open('D:/Chungkhoan/data/reports/optimization_2026-06-08T060735.723866Z.json', 'r') as f:
        report = json.load(f)
        
    best_params = report['selected_candidate']['params']
    
    with open('D:/Chungkhoan/configs/rules_t2.json', 'r') as f:
        rules = json.load(f)
        
    for k, v in best_params.items():
        keys = k.split('.')
        if len(keys) == 1:
            rules[keys[0]] = v
        else:
            rules[keys[0]][keys[1]] = v
            
    # Apply user overrides
    rules['time_horizon_days'] = 20
    rules['signal_thresholds']['buy_setup'] = 45
    
    with open('D:/Chungkhoan/configs/rules_t2.json', 'w') as f:
        json.dump(rules, f, indent=2)
        
    print("Successfully applied candidate 66 and user overrides (time_horizon_days=20, buy_setup=45) to rules_t2.json")

if __name__ == '__main__':
    apply_overrides()
