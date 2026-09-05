import os
import sys
import json

# Reconfigure stdout for UTF-8 support on Windows default terminal (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCHOOL_DB_FILE = "school_db.json"


def init_school_db(filepath: str = SCHOOL_DB_FILE):
    """
    school_db.json database file ko initialize karta hai agar exist nahi karti.
    Structure: classes, staff, students
    """
    if not os.path.exists(filepath):
        default_db = {
            "classes": {
                "Class 9": ["Math", "Physics", "Chemistry", "English"],
                "Class 10": ["Math", "Physics", "Biology", "Urdu"]
            },
            "staff": {},
            "students": {}
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(default_db, f, indent=2)
            print(f"[INFO] Initialized new school database: {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize school database: {e}")


def load_school_db(filepath: str = SCHOOL_DB_FILE) -> dict:
    """
    school_db.json file se data load karta hai.
    """
    init_school_db(filepath)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load school database: {e}")
        return {"classes": {}, "staff": {}, "students": {}}


def save_school_db(db_data: dict, filepath: str = SCHOOL_DB_FILE) -> bool:
    """
    school_db.json file mein data save karta hai.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, indent=2)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save school database: {e}")
        return False


# =============================================================
# CLASS MANAGEMENT OPERATIONS
# =============================================================
def get_all_classes(filepath: str = SCHOOL_DB_FILE) -> dict:
    """
    Tamam classes aur unke subjects return karta hai. dict of {class_name: [subjects]}
    """
    db = load_school_db(filepath)
    return db.get("classes", {})


def add_class(class_name: str, subjects: list, filepath: str = SCHOOL_DB_FILE) -> tuple[bool, str]:
    """
    Nayi Class aur uske subjects database mein add karta hai.
    """
    clean_class = class_name.strip()
    if not clean_class:
        return False, "Class name khali nahi ho sakta."
        
    clean_subjects = [s.strip() for s in subjects if s.strip()]
    if not clean_subjects:
        return False, "Kripya kam se kam ek subject zaroor add karein."

    db = load_school_db(filepath)
    classes = db.get("classes", {})

    classes[clean_class] = clean_subjects
    db["classes"] = classes

    if save_school_db(db, filepath):
        return True, f"[SUCCESS] Class '{clean_class}' added with subjects: {', '.join(clean_subjects)}"
    return False, "Database save karne mein error aaya."


def delete_class(class_name: str, filepath: str = SCHOOL_DB_FILE) -> tuple[bool, str]:
    """
    Class ko database se remove karta hai.
    """
    db = load_school_db(filepath)
    classes = db.get("classes", {})

    if class_name in classes:
        del classes[class_name]
        db["classes"] = classes
        if save_school_db(db, filepath):
            return True, f"[SUCCESS] Class '{class_name}' successfully deleted."
        return False, "Failed to save database."
    return False, f"Class '{class_name}' not found."


# =============================================================
# STAFF / TEACHER OPERATIONS
# =============================================================
def get_all_staff(filepath: str = SCHOOL_DB_FILE) -> dict:
    """
    Tamam staff members return karta hai. dict of {username: staff_details}
    """
    db = load_school_db(filepath)
    return db.get("staff", {})


def get_staff(username: str, filepath: str = SCHOOL_DB_FILE) -> dict:
    """
    Specific staff member ki details return karta hai.
    """
    staff_all = get_all_staff(filepath)
    return staff_all.get(username.strip().lower(), None)


def add_staff_record(name: str, username: str, password_hash: str, salt: str, assignments: dict, filepath: str = SCHOOL_DB_FILE) -> tuple[bool, str]:
    """
    Staff record (with hashed password & class/subject assignments) database mein add karta hai.
    """
    clean_user = username.strip().lower()
    clean_name = name.strip()

    if not clean_user or not clean_name:
        return False, "Name aur Username zaroori hain."

    db = load_school_db(filepath)
    staff_dict = db.get("staff", {})

    staff_dict[clean_user] = {
        "name": clean_name,
        "username": clean_user,
        "password_hash": password_hash,
        "salt": salt,
        "assignments": assignments  # Format: {"Class 9": ["Math", "Physics"], ...}
    }

    db["staff"] = staff_dict
    if save_school_db(db, filepath):
        return True, f"[SUCCESS] Staff member '{clean_name}' (@{clean_user}) successfully created."
    return False, "Database save error."


def delete_staff(username: str, filepath: str = SCHOOL_DB_FILE) -> tuple[bool, str]:
    """
    Staff member ko database se remove karta hai.
    """
    clean_user = username.strip().lower()
    db = load_school_db(filepath)
    staff_dict = db.get("staff", {})

    if clean_user in staff_dict:
        staff_name = staff_dict[clean_user].get("name", clean_user)
        del staff_dict[clean_user]
        db["staff"] = staff_dict
        if save_school_db(db, filepath):
            return True, f"[SUCCESS] Staff member '{staff_name}' (@{clean_user}) deleted."
        return False, "Database save error."
    return False, f"Staff username '@{clean_user}' not found."


# =============================================================
# STUDENT OPERATIONS
# =============================================================
def get_all_students(filepath: str = SCHOOL_DB_FILE) -> dict:
    """
    Tamam registered students return karta hai.
    """
    db = load_school_db(filepath)
    return db.get("students", {})


def get_students_by_class(class_name: str, filepath: str = SCHOOL_DB_FILE) -> dict:
    """
    Specific Class ke tamam students return karta hai. dict of {student_key: student_info}
    """
    students_all = get_all_students(filepath)
    return {
        k: v for k, v in students_all.items()
        if v.get("class_name") == class_name
    }


def add_student_record(name: str, roll_number: str, class_name: str, photo_path: str, filepath: str = SCHOOL_DB_FILE) -> tuple[bool, str]:
    """
    Student Record DB mein save karta hai. Student Key = "<Class>_<RollNumber>"
    """
    clean_name = name.strip()
    clean_roll = roll_number.strip()
    clean_class = class_name.strip()

    if not clean_name or not clean_roll or not clean_class:
        return False, "Name, Roll Number aur Class sab zaroori hain."

    student_key = f"{clean_class}_{clean_roll}"
    db = load_school_db(filepath)
    students = db.get("students", {})

    students[student_key] = {
        "student_key": student_key,
        "name": clean_name,
        "roll_number": clean_roll,
        "class_name": clean_class,
        "photo_path": photo_path
    }

    db["students"] = students
    if save_school_db(db, filepath):
        return True, f"[SUCCESS] Student '{clean_name}' (Roll: {clean_roll}, {clean_class}) saved to database."
    return False, "Database save error."


def delete_student_record(class_name: str, roll_number: str, filepath: str = SCHOOL_DB_FILE) -> tuple[bool, str]:
    """
    Student record key "<Class>_<RollNumber>" ko database se delete karta hai.
    """
    student_key = f"{class_name.strip()}_{roll_number.strip()}"
    db = load_school_db(filepath)
    students = db.get("students", {})

    if student_key in students:
        s_name = students[student_key].get("name", student_key)
        del students[student_key]
        db["students"] = students
        if save_school_db(db, filepath):
            return True, f"[SUCCESS] Student '{s_name}' (Roll: {roll_number}) deleted from database."
        return False, "Database save error."
    return False, f"Student key '{student_key}' not found."
