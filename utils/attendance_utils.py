import os
import sys
import csv
from datetime import datetime
import pandas as pd

# Reconfigure stdout for UTF-8 support on Windows default terminal (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ATTENDANCE_CSV_FILE = "attendance.csv"
CSV_HEADERS = ["Name", "Roll_Number", "Class", "Subject", "Date", "Time", "Marked_By_Teacher"]


# -------------------------------------------------------------
# 1. INITIALIZE ATTENDANCE CSV WITH NEW SCHEMA
# -------------------------------------------------------------
def init_attendance_file(filepath: str = ATTENDANCE_CSV_FILE):
    """
    attendance.csv file setup karta hai with schema:
    [Name, Roll_Number, Class, Subject, Date, Time, Marked_By_Teacher]
    """
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        print(f"[INFO] Created new attendance file: {filepath}")


# -------------------------------------------------------------
# 2. DUPLICATE ATTENDANCE CHECK (SCOPED BY CLASS & SUBJECT)
# -------------------------------------------------------------
def is_already_marked_scoped(roll_number: str, class_name: str, subject: str, filepath: str = ATTENDANCE_CSV_FILE, check_date: str = None) -> bool:
    """
    Check karta hai ke kya student ki attendance specified (Class + Subject + Date) ke liye pehle se lagi hai.
    """
    if check_date is None:
        check_date = datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return False

    clean_roll = str(roll_number).strip().lower()
    clean_class = str(class_name).strip().lower()
    clean_subject = str(subject).strip().lower()

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 5:
                    r_no = row[1].strip().lower() if len(row) > 1 else ""
                    c_name = row[2].strip().lower() if len(row) > 2 else ""
                    s_name = row[3].strip().lower() if len(row) > 3 else ""
                    d_str = row[4].strip() if len(row) > 4 else ""

                    if r_no == clean_roll and c_name == clean_class and s_name == clean_subject and d_str == check_date:
                        return True
    except Exception as e:
        print(f"[WARN] Error checking duplicate attendance: {e}")
        
    return False


# -------------------------------------------------------------
# 3. MARK ATTENDANCE FUNCTION (CLASS & SUBJECT SCOPED)
# -------------------------------------------------------------
def mark_attendance_scoped(name: str, roll_number: str, class_name: str, subject: str, teacher_name: str, filepath: str = ATTENDANCE_CSV_FILE):
    """
    Student ki attendance specified Class, Subject aur Teacher ke against mark karta hai.
    """
    init_attendance_file(filepath)

    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    # Duplicate check
    if is_already_marked_scoped(roll_number, class_name, subject, filepath, today):
        return False, f"⚠️ '{name}' (Roll: {roll_number}) ki attendance {class_name} - {subject} ke liye aaj ({today}) pehle se lag chuki hai."

    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([name, roll_number, class_name, subject, today, time_now, teacher_name])
        return True, f"✅ Attendance marked for '{name}' (Roll: {roll_number}) | {class_name} - {subject} at {time_now}."
    except Exception as e:
        return False, f"❌ Error writing attendance to CSV: {str(e)}"


# -------------------------------------------------------------
# 4. VIEW & FILTER ATTENDANCE RECORDS (SAFE & GRACEFUL)
# -------------------------------------------------------------
def get_attendance_records(filepath: str = ATTENDANCE_CSV_FILE, date_filter=None, class_filter: str = None, subject_filter: str = None, teacher_filter: str = None, name_filter: str = None):
    """
    Attendance records load karke multi-level filtering apply karta hai.
    Khali ya corrupted CSV files par crash hone ke bajaye graceful empty DataFrame return karta hai.
    Returns: (filtered_dataframe, metrics_dict)
    """
    init_attendance_file(filepath)

    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            df = pd.read_csv(filepath)
        else:
            df = pd.DataFrame(columns=CSV_HEADERS)
    except Exception as e:
        print(f"[WARN] Could not read CSV file {filepath}: {e}")
        df = pd.DataFrame(columns=CSV_HEADERS)

    # Ensure all required columns exist without index mismatch crash
    for col in CSV_HEADERS:
        if col not in df.columns:
            if len(df) > 0:
                df[col] = ""
            else:
                df[col] = pd.Series(dtype='object')

    # If df is empty, return structured empty DataFrame
    if df.empty:
        df = pd.DataFrame(columns=CSV_HEADERS)

    today_str = datetime.now().strftime("%Y-%m-%d")

    total_entries = len(df)
    today_df = df[df["Date"].astype(str) == today_str] if (not df.empty and "Date" in df.columns) else pd.DataFrame()
    today_count = len(today_df)
    unique_today = today_df["Roll_Number"].nunique() if (not today_df.empty and "Roll_Number" in today_df.columns) else 0

    filtered_df = df.copy()

    if not filtered_df.empty:
        if date_filter:
            if isinstance(date_filter, str):
                target_date = date_filter
            elif hasattr(date_filter, 'strftime'):
                target_date = date_filter.strftime("%Y-%m-%d")
            else:
                target_date = str(date_filter)
            filtered_df = filtered_df[filtered_df["Date"].astype(str) == target_date]

        if class_filter and class_filter.strip() and class_filter != "All Classes":
            filtered_df = filtered_df[filtered_df["Class"].astype(str).str.lower() == class_filter.strip().lower()]

        if subject_filter and subject_filter.strip() and subject_filter != "All Subjects":
            filtered_df = filtered_df[filtered_df["Subject"].astype(str).str.lower() == subject_filter.strip().lower()]

        if teacher_filter and teacher_filter.strip():
            filtered_df = filtered_df[filtered_df["Marked_By_Teacher"].astype(str).str.lower().str.contains(teacher_filter.strip().lower(), na=False)]

        if name_filter and name_filter.strip():
            search_term = name_filter.strip().lower()
            filtered_df = filtered_df[
                filtered_df["Name"].astype(str).str.lower().str.contains(search_term, na=False) |
                filtered_df["Roll_Number"].astype(str).str.lower().str.contains(search_term, na=False)
            ]

    metrics = {
        "total_entries": total_entries,
        "today_count": today_count,
        "unique_today": unique_today,
        "filtered_count": len(filtered_df)
    }

    return filtered_df, metrics
