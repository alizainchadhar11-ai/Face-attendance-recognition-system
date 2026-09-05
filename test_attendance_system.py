import os
import sys
import io
import pickle
import shutil
import json
import numpy as np

# UTF-8 stdout fix for Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.db_utils import (
    init_school_db,
    add_class,
    delete_class,
    add_staff_record,
    delete_staff,
    add_student_record,
    delete_student_record,
    get_all_classes,
    get_students_by_class
)
from utils.auth_utils import (
    is_first_time_setup,
    setup_admin_password,
    verify_admin_password,
    create_staff_account,
    verify_staff_password
)
from utils.face_utils import (
    cosine_distance,
    delete_student_classwise,
    recognize_face_scoped
)
from utils.attendance_utils import (
    init_attendance_file,
    is_already_marked_scoped,
    mark_attendance_scoped,
    get_attendance_records,
    CSV_HEADERS
)
from utils.report_utils import (
    bulk_import_students_from_csv,
    get_student_attendance_summary,
    get_absentee_list,
    generate_excel_report
)


def test_db_utils():
    print("Testing db_utils...")
    test_db_file = "test_school_db.json"
    if os.path.exists(test_db_file):
        os.remove(test_db_file)

    init_school_db(test_db_file)
    assert os.path.exists(test_db_file), "Database file should exist"

    # Test Add Class
    c_ok, c_msg = add_class("Class 9", ["Math", "Physics"], test_db_file)
    assert c_ok == True, f"Add class failed: {c_msg}"
    classes = get_all_classes(test_db_file)
    assert "Class 9" in classes, "Class 9 should be in classes dict"

    # Test Add Staff Record
    s_ok, s_msg = add_staff_record("Sir Ahmed", "sir_ahmed", "hash123", "salt123", {"Class 9": ["Math"]}, test_db_file)
    assert s_ok == True, f"Add staff failed: {s_msg}"

    # Test Add Student Record
    st_ok, st_msg = add_student_record("Ali Zain", "101", "Class 9", "known_faces/Class 9/AliZain_101.jpg", test_db_file)
    assert st_ok == True, f"Add student failed: {st_msg}"
    
    students_c9 = get_students_by_class("Class 9", test_db_file)
    assert "Class 9_101" in students_c9, "Student should be found in Class 9"

    # Cleanup DB
    if os.path.exists(test_db_file):
        os.remove(test_db_file)

    print("PASSED: db_utils test passed!")


def test_auth_system_extended():
    print("Testing auth_utils (Admin + Staff)...")
    test_cfg = "test_admin_config.json"
    test_db = "test_auth_school_db.json"
    
    if os.path.exists(test_cfg): os.remove(test_cfg)
    if os.path.exists(test_db): os.remove(test_db)
    init_school_db(test_db)

    # 1. Admin Setup & Verification
    assert is_first_time_setup(test_cfg) == True
    a_ok, _ = setup_admin_password("admin123", "admin123", test_cfg)
    assert a_ok == True
    assert verify_admin_password("admin123", test_cfg) == True
    assert verify_admin_password("wrongadmin", test_cfg) == False

    # 2. Staff Account Creation & Verification
    st_ok, st_msg = create_staff_account("Sir Ahmed", "sir_ahmed", "teacher123", {"Class 9": ["Math"]}, test_db)
    assert st_ok == True, f"Create staff failed: {st_msg}"

    v_ok, v_msg, v_data = verify_staff_password("sir_ahmed", "teacher123", test_db)
    assert v_ok == True, f"Staff verify failed: {v_msg}"
    assert v_data.get("name") == "Sir Ahmed"

    v_fail, _, _ = verify_staff_password("sir_ahmed", "wrongpwd", test_db)
    assert v_fail == False, "Wrong staff password should fail"

    # Cleanup
    if os.path.exists(test_cfg): os.remove(test_cfg)
    if os.path.exists(test_db): os.remove(test_db)

    print("PASSED: auth_utils test passed!")


def test_scoped_attendance_utils():
    print("Testing attendance_utils (Scoped Schema)...")
    test_csv = "test_attendance_scoped.csv"
    if os.path.exists(test_csv):
        os.remove(test_csv)

    init_attendance_file(test_csv)
    assert os.path.exists(test_csv)

    # 1. Mark attendance first time
    ok1, msg1 = mark_attendance_scoped("Ali Zain", "101", "Class 9", "Math", "Sir Ahmed", test_csv)
    assert ok1 == True, f"First mark failed: {msg1}"

    # 2. Duplicate check
    dup = is_already_marked_scoped("101", "Class 9", "Math", test_csv)
    assert dup == True, "Should be marked"

    # 3. Mark duplicate
    ok2, msg2 = mark_attendance_scoped("Ali Zain", "101", "Class 9", "Math", "Sir Ahmed", test_csv)
    assert ok2 == False, f"Duplicate should be rejected: {msg2}"

    # 4. Filter logs
    df, metrics = get_attendance_records(test_csv, class_filter="Class 9", subject_filter="Math")
    assert len(df) == 1
    assert df.iloc[0]["Name"] == "Ali Zain"
    assert str(df.iloc[0]["Roll_Number"]) == "101"
    assert df.iloc[0]["Class"] == "Class 9"
    assert df.iloc[0]["Subject"] == "Math"
    assert df.iloc[0]["Marked_By_Teacher"] == "Sir Ahmed"

    if os.path.exists(test_csv):
        os.remove(test_csv)

    print("PASSED: attendance_utils test passed!")


def test_empty_and_corrupted_csv():
    print("Testing empty & corrupted CSV handling in get_attendance_records...")
    empty_csv = "test_empty_attendance.csv"
    
    # Case A: 0-byte file
    with open(empty_csv, "w") as f:
        pass
    df, metrics = get_attendance_records(empty_csv)
    assert df.empty == True, "Empty CSV should return empty DataFrame gracefully"
    assert metrics["total_entries"] == 0
    assert list(df.columns) == CSV_HEADERS
    
    # Case B: Legacy 3-column CSV with 2 rows
    with open(empty_csv, "w", encoding="utf-8") as f:
        f.write("Name,Date,Time\nAli,2026-08-31,10:00:00\nSara,2026-08-31,10:05:00\n")
    df_legacy, metrics_legacy = get_attendance_records(empty_csv)
    assert len(df_legacy) == 2, "Legacy CSV should load 2 rows without crash"
    assert "Class" in df_legacy.columns and "Subject" in df_legacy.columns
    
    if os.path.exists(empty_csv):
        os.remove(empty_csv)
        
    print("PASSED: empty & corrupted CSV test passed!")


def test_phase2_reports_and_bulk_import():
    print("Testing Phase 2 (Bulk Import, Reports, Absentees, Excel)...")
    
    delete_student_record("Class 9", "501")
    delete_student_record("Class 9", "502")

    # 1. Test Bulk Import
    csv_content = "Name,Roll_Number,Class\nImport Student 1,501,Class 9\nImport Student 2,502,Class 9\n"
    csv_bytes = io.BytesIO(csv_content.encode("utf-8"))
    
    imp_count, skip_count, msg = bulk_import_students_from_csv(csv_bytes, "test_known_faces")
    assert imp_count >= 1, f"Bulk import failed: {msg}"

    # 2. Test Student Attendance Summary
    summary_df = get_student_attendance_summary(class_filter="Class 9")
    assert summary_df is not None, "Summary should return DataFrame"

    # 3. Test Absentee List Extraction
    absentee_df = get_absentee_list(class_filter="Class 9")
    assert absentee_df is not None, "Absentees should be extracted"

    # 4. Test Excel Report Generation
    excel_bytes = generate_excel_report(summary_df, absentee_df, summary_df)
    assert len(excel_bytes) > 0, "Excel output should be non-empty bytes"
    assert excel_bytes.startswith(b"PK"), "Excel .xlsx file should start with PK zip header"

    delete_student_record("Class 9", "501")
    delete_student_record("Class 9", "502")
    if os.path.exists("test_known_faces"):
        shutil.rmtree("test_known_faces", ignore_errors=True)

    print("PASSED: Phase 2 reports & bulk import test passed!")


if __name__ == "__main__":
    test_db_utils()
    test_auth_system_extended()
    test_scoped_attendance_utils()
    test_empty_and_corrupted_csv()
    test_phase2_reports_and_bulk_import()
    print("\nALL PHASE 1 & PHASE 2 UNIT TESTS PASSED SUCCESSFULLY!")
