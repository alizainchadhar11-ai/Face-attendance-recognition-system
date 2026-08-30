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

# -------------------------------------------------------------
# 1. INITIALIZE ATTENDANCE CSV
# Attendance file agar exist nahi karti to header ke saath create karta hai
# -------------------------------------------------------------
def init_attendance_file(filepath: str = "attendance.csv"):
    """
    attendance.csv file ko setup karta hai. Header: ["Name", "Date", "Time"]
    """
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])
        print(f"[INFO] Created new attendance file: {filepath}")


# -------------------------------------------------------------
# 2. DUPLICATE ATTENDANCE CHECK FOR TODAY
# Ek hi din mein same student ki attendance dobara nahi lagne deta
# -------------------------------------------------------------
def is_already_marked(name: str, filepath: str = "attendance.csv", check_date: str = None) -> bool:
    """
    Check karta hai ke kya student ki attendance specified date (default: today) ko pehle se lagi hai.
    """
    if check_date is None:
        check_date = datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(filepath):
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header row
            for row in reader:
                if len(row) >= 2 and row[0].strip().lower() == name.strip().lower() and row[1].strip() == check_date:
                    return True
    except Exception as e:
        print(f"[WARN] Error reading attendance file: {e}")
        
    return False


# -------------------------------------------------------------
# 3. MARK ATTENDANCE FUNCTION
# Student name, Aaj ki Date aur Time CSV mein log karta hai
# -------------------------------------------------------------
def mark_attendance(name: str, filepath: str = "attendance.csv"):
    """
    Student ki attendance mark karta hai agar aaj pehle se nahi lagi.
    Returns: (success_bool, message_string)
    """
    init_attendance_file(filepath)

    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    # Duplicate check
    if is_already_marked(name, filepath, today):
        return False, f"⚠️ '{name}' ki attendance aaj ({today}) pehle se lag chuki hai."

    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([name, today, time_now])
        return True, f"✅ Attendance successfully marked for '{name}' at {time_now} ({today})."
    except Exception as e:
        return False, f"❌ Error writing attendance to file: {str(e)}"


# -------------------------------------------------------------
# 4. VIEW & FILTER ATTENDANCE RECORDS
# Pandas DataFrame Return karta hai table view aur filtering ke liye
# -------------------------------------------------------------
def get_attendance_records(filepath: str = "attendance.csv", date_filter=None, name_filter: str = None):
    """
    Attendance records load karke date aur name filtering apply karta hai.
    Returns: (filtered_dataframe, metrics_dict)
    """
    init_attendance_file(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # Ensure required columns exist
    for col in ["Name", "Date", "Time"]:
        if col not in df.columns:
            df[col] = []

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Metrics before filtering
    total_entries = len(df)
    today_df = df[df["Date"] == today_str] if not df.empty else pd.DataFrame()
    today_count = len(today_df)
    unique_today = today_df["Name"].nunique() if not today_df.empty else 0

    # Apply filters
    filtered_df = df.copy()
    
    if date_filter:
        if isinstance(date_filter, str):
            filtered_df = filtered_df[filtered_df["Date"] == date_filter]
        elif hasattr(date_filter, 'strftime'):
            target_date = date_filter.strftime("%Y-%m-%d")
            filtered_df = filtered_df[filtered_df["Date"] == target_date]

    if name_filter and name_filter.strip():
        search_name = name_filter.strip().lower()
        filtered_df = filtered_df[filtered_df["Name"].str.lower().str.contains(search_name, na=False)]

    metrics = {
        "total_entries": total_entries,
        "today_count": today_count,
        "unique_today": unique_today,
        "filtered_count": len(filtered_df)
    }

    return filtered_df, metrics
