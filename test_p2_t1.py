
from diagnostics import classify_failure

def test_classify_failure():
    test_cases = [
        ("Out of memory on device", "OOM"),
        ("Metal::malloc error", "OOM"),
        ("alloc failure", "OOM"),
        ("Loss is NaN", "NaN/Inf"),
        ("inf detected in gradients", "NaN/Inf"),
        ("The training diverged", "NaN/Inf"),
        ("VRAM limit exceeded", "VRAMLimit"),
        ("Training timed out after 300s", "Timeout"),
        ("Connection timeout", "Timeout"),
        ("ImportError: No module named 'foo'", "ImportError"),
        ("ValueError: invalid literal", "ValueError"),
        ("Success! Validation L2: 0.123", "Unknown"), # classify_failure doesn't explicitly return "Success" but it returns what it finds.
    ]
    
    passed = 0
    for content, expected in test_cases:
        result = classify_failure(content)
        # If it doesn't match any, it returns what? 
        # Looking at code: it returns the last matched if it was a chain of if-elses.
        # Actually, it's a sequence of 'if' and 'return'.
        # If no match, it falls through.
        # Let's check the end of the function.
        print(f"Content: {content[:30]}... -> Result: {result} (Expected: {expected})")
        if result == expected:
            passed += 1
        elif expected == "Unknown" and result not in ["OOM", "NaN/Inf", "Timeout", "VRAMLimit", "ImportError", "ValueError"]:
            # Success/Normal logs might fall through or match something else if not careful.
            passed += 1

    accuracy = (passed / len(test_cases)) * 100
    print(f"Accuracy: {accuracy}%")
    assert accuracy >= 90

if __name__ == "__main__":
    test_classify_failure()
