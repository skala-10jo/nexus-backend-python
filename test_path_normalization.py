"""
Test script to verify path normalization works correctly
"""

def normalize_file_path(file_path: str) -> str:
    """
    Normalize file path to relative path.
    """
    # If path contains 'uploads/documents' or 'uploads\\documents', extract relative part
    if 'uploads/documents' in file_path:
        # Handle forward slashes (Unix/macOS)
        parts = file_path.split('uploads/documents/')
        if len(parts) > 1:
            return parts[-1]
    elif 'uploads\\documents' in file_path:
        # Handle backslashes (Windows)
        parts = file_path.split('uploads\\documents\\')
        if len(parts) > 1:
            # Convert Windows backslashes to forward slashes
            return parts[-1].replace('\\', '/')
    
    # Already a relative path, return as-is
    return file_path


# Test cases
test_cases = [
    # Relative path (should stay the same)
    ("2025/12/02/8b0df19a-033f-4013-b9df-378f556e0684.txt", 
     "2025/12/02/8b0df19a-033f-4013-b9df-378f556e0684.txt"),
    
    # macOS absolute path
    ("/Users/SKAX/Desktop/SKALA-TP/final_pj/nexus/backend-java/uploads/documents/2025/12/02/8b0df19a-033f-4013-b9df-378f556e0684.txt",
     "2025/12/02/8b0df19a-033f-4013-b9df-378f556e0684.txt"),
    
    # WSL absolute path
    ("/home/sehui/skala-final/nexus/backend-java/uploads/documents/2025/12/02/8b0df19a-033f-4013-b9df-378f556e0684.txt",
     "2025/12/02/8b0df19a-033f-4013-b9df-378f556e0684.txt"),
    
    # Windows absolute path
    ("C:\\Users\\SKAX\\Desktop\\uploads\\documents\\2025\\12\\02\\file.txt",
     "2025/12/02/file.txt"),
]

print("=" * 80)
print("Path Normalization Test")
print("=" * 80)

all_passed = True
for input_path, expected_output in test_cases:
    result = normalize_file_path(input_path)
    passed = result == expected_output
    all_passed = all_passed and passed
    
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status}")
    print(f"  Input:    {input_path}")
    print(f"  Expected: {expected_output}")
    print(f"  Got:      {result}")

print("\n" + "=" * 80)
if all_passed:
    print("✅ All tests passed!")
else:
    print("❌ Some tests failed!")
print("=" * 80)
