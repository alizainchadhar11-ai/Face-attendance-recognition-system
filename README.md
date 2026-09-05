# 🏫 School & Institute Management System (Phase 1)

Yeh ek complete, modular aur automatic **School / Institute Management System** hai jo `facenet-pytorch` (`MTCNN` + `InceptionResnetV1` pretrained on `vggface2`) aur `Streamlit` web portal use karta hai.

Single-folder system se upgrade karke isay ek full-fledged **Role-Based School Portal** banaya gaya hai.

---

## 🌟 Key Features (Phase 1)

1. **Role-Based Authentication Portal:**
   - **Admin Portal**: First-time password setup aur secure login (`admin_config.json` PBKDF2 hashing ke sath).
   - **Staff / Teacher Portal**: Staff member login (`Username` + `Hashed Password`). Accounts exclusively **Admin** create karta hai. Staff self-signup nahi kar sakte.
   - Global **Logout** button.

2. **Structured JSON Database & Class-Wise Data Organization:**
   - `school_db.json`: JSON database managing **Classes** (with subjects list), **Staff Accounts** (with PBKDF2 hashed passwords and assigned classes/subjects), and **Student Records** (Name, Roll Number, Class).
   - **Class-Wise Photos**: Saved in `known_faces/Class_9/`, `known_faces/Class_10/`, etc.
   - **Class-Scoped Embeddings (`embeddings.pkl`)**: Face embeddings dictionary organized by Class Name. When a teacher takes attendance for Class 9, recognition matches **only** against Class 9 students, eliminating cross-class false matches and accelerating speed!
   - **Upgraded Attendance CSV Schema**: `[Name, Roll_Number, Class, Subject, Date, Time, Marked_By_Teacher]`.

3. **Admin Dashboard (Admin Role):**
   - **Tab 1: 🏫 Manage Classes**: Add new classes with custom subject lists (e.g. Class 9: Math, Physics, Chemistry, English); delete classes.
   - **Tab 2: 👨‍🏫 Manage Staff**: Create staff accounts, set credentials, assign specific classes & subjects per class; view staff list; delete staff.
   - **Tab 3: 🎓 Manage Students**: Register student (Upload/Capture photo + Name + Roll Number + Class Dropdown), view enrolled students filtered by class, delete student.
   - **Tab 4: 📊 All Attendance Logs**: View global logs, filter by Class, Subject, Date, Teacher, or Student Name/Roll Number + CSV Export.

4. **Staff/Teacher Dashboard (Staff Role):**
   - **Step 1**: Select assigned Class (e.g., Class 9).
   - **Step 2**: Select assigned Subject for that class (e.g., Math).
   - **Step 3**: Launch Face Recognition Attendance for Class 9 - Math:
     - Live webcam capture -> MTCNN face detection -> InceptionResnetV1 embedding.
     - Matches face **only** against Class 9 students.
     - Auto-logs attendance under `[Name, Roll_Number, Class 9, Math, Date, Time, Teacher_Name]`.
     - Rejects duplicate attendance for the same student + subject on the same calendar day.

---

## 📁 Project Structure

```text
face_attendance_app/
│
├── app.py                   # Main Streamlit UI Portal (Role Router: Login, Admin Dash, Staff Dash)
├── known_faces/             # Class-wise reference photo folders (known_faces/Class_9/, etc.)
├── embeddings.pkl           # Class-wise face embeddings cache { "Class 9": { "101": {...} } }
├── attendance.csv           # Upgraded CSV log (Name, Roll_Number, Class, Subject, Date, Time, Marked_By_Teacher)
├── school_db.json           # School JSON database (Classes, Staff Accounts, Student Records)
├── admin_config.json        # Admin hashed credentials file
├── test_attendance_system.py# Phase 1 unit test suite
├── utils/
│   ├── __init__.py
│   ├── db_utils.py          # JSON Database manager for Classes, Staff & Students
│   ├── auth_utils.py        # Admin & Staff PBKDF2 hashing, account creation & login verification
│   ├── face_utils.py        # MTCNN, InceptionResnetV1, class-wise embeddings & scoped face matching
│   └── attendance_utils.py  # Upgraded CSV attendance logging & multi-filter records retrieval
├── requirements.txt         # Project dependencies
└── README.md                # System documentation
```

---

## 🚀 How to Run the App (Step-by-Step)

### Step 1: Open Terminal / Command Prompt
Project folder mein terminal kholein:
```bash
cd "c:\Users\DELL\Desktop\Face attendance recognition system"
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application
```bash
streamlit run app.py
```
App browser mein auto-open ho jaye gi (`http://localhost:8501`).

---

## 📖 How to Use the App

### 1️⃣ Admin Login & Setup
- First-time setup: Admin Portal choose karke Admin Password set karein.
- Admin Dashboard mein:
  1. **Manage Classes**: Pehle Classes (e.g. `Class 9`) aur unke Subjects (e.g. `Math, Physics`) add karein.
  2. **Manage Staff**: Teachers add karein aur unhe Classes/Subjects assign karein.
  3. **Manage Students**: Students ki photo upload karke unka Name, Roll Number, aur Class assign karke register karein.

### 2️⃣ Staff / Teacher Login & Attendance
- Teacher Portal select karein aur Username & Password daal kar login karein.
- Assigned Class select karein -> Assigned Subject select karein -> Webcam se photo capture karke attendance mark karein.
- System automatically Class, Subject, aur Teacher Name ke sath attendance log kar dega!
