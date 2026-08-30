# 👤 AI Face Recognition Attendance System (Local Production App)

Yeh ek complete, clean aur automatic **Face Recognition based Attendance System** hai jo `facenet-pytorch` (`MTCNN` for detection + `InceptionResnetV1` pretrained on `vggface2` for embeddings) aur `Streamlit` web interface use karta hai.

Google Colab notebook ke cell-by-cell code ko ek proper modular, local production app mein convert kiya gaya hai.

---

## 🌟 Key Features

1. **First-Time Setup & Secure Admin Authentication:**
   - Pehli baar app chalane par "First-Time Admin Password Setup" screen aati hai jahan Admin apna custom password set karta hai.
   - Password plain text mein save **nahi** hota. Security ke liye `hashlib.pbkdf2_hmac` (SHA-256 + 16-byte random salt, 100,000 iterations) se hash karke `admin_config.json` mein save hota hai.
   - Student Registration aur Student Management/Deletion features sirf Admin password se protected hain.

2. **Public Attendance & Log Viewing:**
   - **Take Attendance** (Webcam capture) aur **View Attendance Logs** public access ke liye khule hain taake koi bhi bina login kiye attendance mark aur logs dekhein sake.

3. **Automatic Student Registration:**
   - Student ki photo kisi bhi filename se upload/capture karein aur student ka naam likhein.
   - System khud face detect karta hai, file ko clean student name se `known_faces/` folder mein save karta hai, aur embedding generate karke `embeddings.pkl` mein store karta hai.
   
4. **Incremental Embedding Cache (`embeddings.pkl`):**
   - Sabhi registered students ki face embeddings `embeddings.pkl` file mein save hoti hain.
   - Jab bhi naya student register hota hai ya app restart hoti hai, purani embeddings dobara compute nahi hoti. Sirf naye students ki embedding calculate hoti hai.

5. **Live Webcam Attendance Capture:**
   - Browser ke direct webcam (`st.camera_input`) se live photo capture karein.
   - MTCNN face crop karke InceptionResnetV1 se cosine distance (threshold = 0.5) calculate karta hai.
   - Match milne par `attendance.csv` (Name, Date, Time) mein automatically attendance log ho jaati hai.

6. **Duplicate Attendance Check:**
   - Same student ki ek hi din (`YYYY-MM-DD`) mein dobara attendance mark nahi hoti. System "Already Marked Today" ka clear notification dikhata hai.

7. **Student Management & Deletion:**
   - Admin dropdown list se kisi bhi student ko choose karke single click par database (`embeddings.pkl`) aur reference photo (`known_faces/`) dono se delete kar sakta hai.

8. **Attendance Analytics & CSV Export:**
   - Attendance records ko interactive table mein dekhein.
   - Date picker aur Name search se records filter karein.
   - Filtered report ko **CSV Download Button** se export karein.

---

## 📁 Project Structure

```text
face_attendance_app/
│
├── app.py                   # Main Streamlit UI application with tabs & auth
├── known_faces/             # Folder containing student reference photos (<StudentName>.jpg)
├── embeddings.pkl           # Cached dictionary of face embeddings {name: vector}
├── attendance.csv           # Attendance log file (Columns: Name, Date, Time)
├── admin_config.json        # Securely hashed Admin Password & Salt file (Auto-created)
├── test_attendance_system.py# Automated unit test suite
├── utils/
│   ├── __init__.py
│   ├── face_utils.py        # Model loading, MTCNN detection, embedding & recognition logic
│   ├── attendance_utils.py  # Attendance marking, duplicate checking & CSV filtering
│   └── auth_utils.py        # Secure admin password setup, PBKDF2 hashing & verification
├── requirements.txt         # Required dependencies
└── README.md                # Project documentation
```

---

## 🔑 Admin Password Reset Guide

Agar aap Admin Password bhool jaate hain ya dobara naya password set karna chahte hain:

1. Project folder se `admin_config.json` file ko delete kar dein:
   ```bash
   # Windows Command Prompt / PowerShell:
   del admin_config.json
   ```
2. App ko dobara run karein (`streamlit run app.py`).
3. System detect karega ke config file missing hai aur **First-Time Admin Password Setup** screen dobara open ho jaye gi.

---

## 🚀 How to Run the App (Step-by-Step)

### Step 1: Open Terminal / Command Prompt
Project folder mein terminal kholein:
```bash
cd "c:\Users\DELL\Desktop\Face attendance recognition system"
```

### Step 2: Create & Activate Virtual Environment (Recommended)
```bash
# Virtual environment create karein
python -m venv venv

# Windows Command Prompt / PowerShell mein activate karein:
.\venv\Scripts\activate

# Linux / Mac mein activate karein:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```bash
streamlit run app.py
```
App browser mein auto-open ho jaye gi (`http://localhost:8501`).

---

## 📖 How to Use the App

### 1️⃣ First-Time Setup & Admin Login
- Pehli baar app chalane par Tab 2 ("Register New Student") par ja kar Admin Password set karein.
- Password set hone ke baad aap automatically log in ho jayein ge.
- Agli baar sirf set kiya hua password daal kar login karna hoga.

### 2️⃣ Register New Student (Admin Only)
- Student ki photo upload karein ya webcam se capture karein.
- Student ka Full Name (e.g. `Ali Zain`) enter karein.
- **"Save & Register Student"** press karein.

### 3️⃣ Manage / Delete Students (Admin Only)
- Dropdown list se student select karein aur **"Delete Student"** press karein.
- Student `embeddings.pkl` aur `known_faces/` folder dono se remove ho jaye ga.

### 4️⃣ Take Attendance (Public Access)
- Webcam ke samne khare hon aur **"Take Photo"** click karein.
- System face recognize karke `attendance.csv` mein attendance log kar de ga.

### 5️⃣ View & Download Logs (Public Access)
- Live metrics, date filter, name search use karein.
- **"Download Filtered Attendance CSV"** button se report download karein.
