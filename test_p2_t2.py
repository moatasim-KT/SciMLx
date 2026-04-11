
from diagnostics import get_fix_strategies

def test_fix_strategies():
    test_cases = [
        ("OOM", {"batch_size": 32}, {"batch_size": 16}),
        ("NaN/Inf", {"lr": 0.001}, {"lr": 0.0001, "grad_clip": 5.0}),
        ("VRAMLimit", {"hidden_dim": 64, "n_layers": 4, "batch_size": 32}, {"hidden_dim": 32, "n_layers": 2, "batch_size": 16}),
        ("Timeout", {"hidden_dim": 64, "n_layers": 4}, {"hidden_dim": 32, "n_layers": 2}),
    ]
    
    for crash_type, config, expected_overrides in test_cases:
        fixes = get_fix_strategies(crash_type, config)
        # We assume for these basic cases there's only 1 fix strategy in the list
        _, result_overrides = fixes[0]
        
        print(f"Crash: {crash_type} -> Result Overrides: {result_overrides} (Expected: {expected_overrides})")
        for key, value in expected_overrides.items():
            assert result_overrides[key] == value
    
    print("All fix strategy tests passed!")

if __name__ == "__main__":
    test_fix_strategies()
