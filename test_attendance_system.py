import os
import sys
import pickle
import shutil
import json
import numpy as np

# UTF-8 stdout fix for Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.attendance_utils import (
    init_attendance_file,
    is_already_marked,
    mark_attendance,
    get_attendance_records
)
from utils.face_utils import cosine_distance, delete_student
from utils.auth_utils import (
    is_first_time_setup,
    setup_admin_password,
    verify_admin_password,
    reset_admin_config
)

def test_cosine_distance():
    print("Testing cosine_distance...")
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    
    assert abs(cosine_distance(v1, v2) - 0.0) < 1e-5, "Identical vectors should have 0 distance"
    assert abs(cosine_distance(v1, v3) - 1.0) < 1e-5, "Orthogonal vectors should have 1.0 distance"
    print("PASSED: Cosine distance test passed!")

def test_attendance_utils():
    print("Testing attendance_utils...")
    test_csv = "test_attendance.csv"
    if os.path.exists(test_csv):
        os.remove(test_csv)
        
    init_attendance_file(test_csv)
    assert os.path.exists(test_csv), "CSV file should be created"
    
    # 1. Mark attendance first time
    success, msg = mark_attendance("Ali Zain", test_csv)
    assert success == True, f"First mark should succeed: {msg}"
    print("Mark 1:", msg)
    
    # 2. Duplicate check
    already = is_already_marked("Ali Zain", test_csv)
    assert already == True, "Should be already marked"
    
    # 3. Mark attendance second time (Duplicate)
    success2, msg2 = mark_attendance("Ali Zain", test_csv)
    assert success2 == False, f"Second mark should be rejected: {msg2}"
    print("Mark 2 (Duplicate):", msg2)
    
    # 4. Get records
    df, metrics = get_attendance_records(test_csv)
    assert len(df) == 1, "Should have exactly 1 record"
    assert df.iloc[0]["Name"] == "Ali Zain"
    assert metrics["today_count"] == 1
    
    # Clean up test file
    if os.path.exists(test_csv):
        os.remove(test_csv)
        
    print("PASSED: Attendance utils test passed!")

def test_delete_student_feature():
    print("Testing delete_student...")
    test_dir = "test_known_faces"
    test_pkl = "test_embeddings.pkl"
    
    os.makedirs(test_dir, exist_ok=True)
    dummy_img = os.path.join(test_dir, "TestDummy.jpg")
    with open(dummy_img, "w") as f:
        f.write("dummy image content")
        
    dummy_data = {"TestDummy": np.zeros(512)}
    with open(test_pkl, "wb") as f:
        pickle.dump(dummy_data, f)
        
    success, msg = delete_student("TestDummy", test_dir, test_pkl)
    assert success == True, f"Deletion failed: {msg}"
    assert not os.path.exists(dummy_img), "Image file should be deleted from known_faces"
    
    with open(test_pkl, "rb") as f:
        data = pickle.load(f)
    assert "TestDummy" not in data, "Student should be removed from embeddings pickle"
    
    # Cleanup
    if os.path.exists(test_pkl):
        os.remove(test_pkl)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    print("PASSED: delete_student test passed!")

def test_auth_system():
    print("Testing admin authentication system...")
    test_config = "test_admin_config.json"
    if os.path.exists(test_config):
        os.remove(test_config)
        
    # 1. First time setup check
    assert is_first_time_setup(test_config) == True, "First time setup should be True when file doesn't exist"
    
    # 2. Validation: Mismatched password
    ok, msg = setup_admin_password("secret123", "different123", test_config)
    assert ok == False, "Mismatch should fail"
    
    # 3. Validation: Short password
    ok, msg = setup_admin_password("123", "123", test_config)
    assert ok == False, "Short password should fail"
    
    # 4. Valid setup
    ok, msg = setup_admin_password("mySecretPass123", "mySecretPass123", test_config)
    assert ok == True, f"Setup failed: {msg}"
    assert os.path.exists(test_config), "Config file should be created"
    
    # 5. Check hashing security (plain text password should NOT be in file)
    with open(test_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert "mySecretPass123" not in json.dumps(cfg), "Plain text password MUST NOT be stored"
    assert "password_hash" in cfg and "salt" in cfg, "File must store password_hash and salt"
    
    # 6. Verify password: Wrong password
    assert verify_admin_password("wrongPass", test_config) == False, "Wrong password should fail"
    
    # 7. Verify password: Correct password
    assert verify_admin_password("mySecretPass123", test_config) == True, "Correct password should pass"
    
    # 8. Reset config test
    reset_ok = reset_admin_config(test_config)
    assert reset_ok == True, "Reset should succeed"
    assert is_first_time_setup(test_config) == True, "After reset, first_time_setup should be True again"
    
    print("PASSED: Admin Auth System test passed!")

if __name__ == "__main__":
    test_cosine_distance()
    test_attendance_utils()
    test_delete_student_feature()
    test_auth_system()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
