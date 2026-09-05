import os
import sys
import io
import csv
from datetime import datetime, timedelta
import pandas as pd
import openpyxl

from utils.db_utils import (
    load_school_db,
    get_all_classes,
    get_all_students,
    get_students_by_class,
    add_student_record
)
from utils.attendance_utils import (
    get_attendance_records,
    ATTENDANCE_CSV_FILE
)

# Reconfigure stdout for UTF-8 support on Windows default terminal (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# =============================================================
# 1. BULK STUDENT CSV IMPORT FUNCTION
# =============================================================
def bulk_import_students_from_csv(csv_file_or_bytes, known_faces_dir: str = "known_faces") -> tuple[int, int, str]:
    """
    CSV file (columns: Name, Roll_Number, Class) se multiple students database mein import karta hai.
    Returns: (imported_count, skipped_count, summary_message)
    """
    try:
        if hasattr(csv_file_or_bytes, "seek"):
            csv_file_or_bytes.seek(0)
            
        df = pd.read_csv(csv_file_or_bytes)
    except Exception as e:
        return 0, 0, f"[ERROR] Could not read CSV file: {str(e)}"

    # Normalize column names
    col_mapping = {}
    for c in df.columns:
        c_clean = str(c).strip().lower().replace(" ", "_")
        if "name" in c_clean or "student" in c_clean:
            col_mapping[c] = "Name"
        elif "roll" in c_clean or "id" in c_clean:
            col_mapping[c] = "Roll_Number"
        elif "class" in c_clean:
            col_mapping[c] = "Class"

    df = df.rename(columns=col_mapping)

    required_cols = ["Name", "Roll_Number", "Class"]
    for req in required_cols:
        if req not in df.columns:
            return 0, 0, f"[ERROR] Missing required CSV column: '{req}'. Expected columns: Name, Roll_Number, Class"

    imported_count = 0
    skipped_count = 0

    all_classes_db = get_all_classes()
    all_students_db = get_all_students()

    for idx, row in df.iterrows():
        s_name = str(row["Name"]).strip() if pd.notna(row["Name"]) else ""
        s_roll = str(row["Roll_Number"]).strip() if pd.notna(row["Roll_Number"]) else ""
        s_class = str(row["Class"]).strip() if pd.notna(row["Class"]) else ""

        if not s_name or not s_roll or not s_class:
            skipped_count += 1
            continue

        # Format roll number without decimals if float
        if s_roll.endswith(".0"):
            s_roll = s_roll[:-2]

        student_key = f"{s_class}_{s_roll}"
        if student_key in all_students_db:
            skipped_count += 1
            continue

        # Create class directory if not exists
        class_folder = os.path.join(known_faces_dir, s_class)
        os.makedirs(class_folder, exist_ok=True)
        placeholder_photo = os.path.join(class_folder, f"{s_roll}_{s_name.replace(' ', '')}.jpg")

        # Save student record in school_db.json
        ok, _ = add_student_record(s_name, s_roll, s_class, placeholder_photo)
        if ok:
            imported_count += 1
        else:
            skipped_count += 1

    msg = f"[SUCCESS] Imported {imported_count} new students. (Skipped: {skipped_count})"
    return imported_count, skipped_count, msg


# =============================================================
# 2. STUDENT-WISE ATTENDANCE PERCENTAGE & SUMMARY
# =============================================================
def get_student_attendance_summary(start_date=None, end_date=None, class_filter: str = None, subject_filter: str = None) -> pd.DataFrame:
    """
    Student-wise total sessions, present days, absent days aur attendance % calculate karta hai.
    Returns: DataFrame [Roll_Number, Name, Class, Present_Days, Absent_Days, Total_Sessions, Attendance_%]
    """
    df_logs, _ = get_attendance_records(
        ATTENDANCE_CSV_FILE,
        date_filter=None,
        class_filter=class_filter,
        subject_filter=subject_filter
    )

    # Date range filtering
    if not df_logs.empty and "Date" in df_logs.columns:
        if start_date:
            s_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else str(start_date)
            df_logs = df_logs[df_logs["Date"].astype(str) >= s_str]
        if end_date:
            e_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else str(end_date)
            df_logs = df_logs[df_logs["Date"].astype(str) <= e_str]

    # Fetch enrolled students from DB
    all_students_db = get_all_students()
    
    rows = []
    for s_key, s_data in all_students_db.items():
        s_class = s_data.get("class_name", "")
        s_roll = str(s_data.get("roll_number", "")).strip()
        s_name = s_data.get("name", "")

        if class_filter and class_filter != "All Classes" and s_class.lower() != class_filter.lower():
            continue

        # Count total unique class/subject sessions held in date range
        if not df_logs.empty:
            class_logs = df_logs[df_logs["Class"].astype(str).str.lower() == s_class.lower()]
            if subject_filter and subject_filter != "All Subjects":
                class_logs = class_logs[class_logs["Subject"].astype(str).str.lower() == subject_filter.lower()]
            
            # Unique session dates/subjects held
            if not class_logs.empty:
                total_sessions = len(class_logs[["Date", "Subject"]].drop_duplicates())
                
                # Student present logs
                student_p = class_logs[class_logs["Roll_Number"].astype(str).str.lower() == s_roll.lower()]
                present_count = len(student_p[["Date", "Subject"]].drop_duplicates())
            else:
                total_sessions = 0
                present_count = 0
        else:
            total_sessions = 0
            present_count = 0

        absent_count = max(0, total_sessions - present_count)
        pct = (present_count / total_sessions * 100.0) if total_sessions > 0 else 100.0

        rows.append({
            "Roll_Number": s_roll,
            "Name": s_name,
            "Class": s_class,
            "Present_Days": present_count,
            "Absent_Days": absent_count,
            "Total_Sessions": total_sessions,
            "Attendance_%": round(pct, 1)
        })

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(by=["Class", "Roll_Number"])
    return result_df


# =============================================================
# 3. CLASS & SUBJECT OVERALL ATTENDANCE % ANALYTICS
# =============================================================
def get_class_subject_analytics(start_date=None, end_date=None, class_filter: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Class-wise aur Subject-wise overall attendance % dataframes return karta hai bar charts ke liye.
    Returns: (class_analytics_df, subject_analytics_df)
    """
    summary_df = get_student_attendance_summary(start_date, end_date, class_filter=class_filter)
    
    if summary_df.empty:
        return pd.DataFrame(columns=["Class", "Attendance_%"]), pd.DataFrame(columns=["Subject", "Attendance_%"])

    # Class-wise average attendance %
    class_df = summary_df.groupby("Class")["Attendance_%"].mean().reset_index()
    class_df["Attendance_%"] = class_df["Attendance_%"].round(1)

    # Subject-wise attendance %
    df_logs, _ = get_attendance_records(ATTENDANCE_CSV_FILE, class_filter=class_filter)
    if not df_logs.empty and "Subject" in df_logs.columns:
        subject_df = df_logs.groupby("Subject")["Name"].count().reset_index()
        subject_df.columns = ["Subject", "Total_Present_Logs"]
    else:
        subject_df = pd.DataFrame(columns=["Subject", "Total_Present_Logs"])

    return class_df, subject_df


# =============================================================
# 4. DAILY ABSENTEE TRACKER
# =============================================================
def get_absentee_list(target_date=None, class_filter: str = None, subject_filter: str = None) -> pd.DataFrame:
    """
    Target date ko un students ki list nikalta hai jo class mein registered hain par attendance log nahi hui.
    Returns: DataFrame [Roll_Number, Name, Class, Subject, Date, Status]
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")
    elif hasattr(target_date, 'strftime'):
        target_date = target_date.strftime("%Y-%m-%d")
    else:
        target_date = str(target_date)

    df_logs, _ = get_attendance_records(ATTENDANCE_CSV_FILE, date_filter=target_date, class_filter=class_filter, subject_filter=subject_filter)

    all_students_db = get_all_students()
    absent_rows = []

    for s_key, s_data in all_students_db.items():
        s_class = s_data.get("class_name", "")
        s_roll = str(s_data.get("roll_number", "")).strip()
        s_name = s_data.get("name", "")

        if class_filter and class_filter != "All Classes" and s_class.lower() != class_filter.lower():
            continue

        # Check if student marked present in CSV on target_date
        is_present = False
        if not df_logs.empty:
            match = df_logs[
                (df_logs["Class"].astype(str).str.lower() == s_class.lower()) &
                (df_logs["Roll_Number"].astype(str).str.lower() == s_roll.lower())
            ]
            if subject_filter and subject_filter != "All Subjects":
                match = match[match["Subject"].astype(str).str.lower() == subject_filter.lower()]

            if not match.empty:
                is_present = True

        if not is_present:
            absent_rows.append({
                "Roll_Number": s_roll,
                "Name": s_name,
                "Class": s_class,
                "Subject": subject_filter if (subject_filter and subject_filter != "All Subjects") else "N/A",
                "Date": target_date,
                "Status": "ABSENT ❌"
            })

    absent_df = pd.DataFrame(absent_rows)
    if not absent_df.empty:
        absent_df = absent_df.sort_values(by=["Class", "Roll_Number"])
    return absent_df


# =============================================================
# 5. MULTI-SHEET EXCEL REPORT EXPORTER (.xlsx)
# =============================================================
def generate_excel_report(summary_df: pd.DataFrame, absentee_df: pd.DataFrame, logs_df: pd.DataFrame) -> bytes:
    """
    Multi-sheet Excel Workbook (.xlsx) generate karta hai and bytes return karta hai.
    Sheet 1: Student Attendance Summary
    Sheet 2: Absentee Report
    Sheet 3: Raw Attendance Logs
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if summary_df is not None and not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='Attendance Summary', index=False)
        else:
            pd.DataFrame({"Info": ["No summary data available"]}).to_excel(writer, sheet_name='Attendance Summary', index=False)

        if absentee_df is not None and not absentee_df.empty:
            absentee_df.to_excel(writer, sheet_name='Absentee Report', index=False)
        else:
            pd.DataFrame({"Info": ["No absentees reported for selected criteria"]}).to_excel(writer, sheet_name='Absentee Report', index=False)

        if logs_df is not None and not logs_df.empty:
            logs_df.to_excel(writer, sheet_name='Raw Attendance Logs', index=False)
        else:
            pd.DataFrame({"Info": ["No attendance logs recorded"]}).to_excel(writer, sheet_name='Raw Attendance Logs', index=False)

    return output.getvalue()
