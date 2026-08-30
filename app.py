import os
import sys
import streamlit as st
from PIL import Image
import pandas as pd

# Reconfigure stdout for UTF-8 support on Windows default terminal (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from utils.face_utils import (
    load_facenet_models,
    load_or_update_embeddings,
    register_student,
    recognize_face,
    delete_student
)
from utils.attendance_utils import (
    init_attendance_file,
    mark_attendance,
    get_attendance_records
)
from utils.auth_utils import (
    is_first_time_setup,
    setup_admin_password,
    verify_admin_password,
    ADMIN_CONFIG_FILE
)

# -------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Face Attendance System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 2. CUSTOM CSS STYLING FOR PREMIUM WOW LOOK & FEEL
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Main container background & font */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    /* Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
    }

    /* Status Badges */
    .status-badge-success {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 16px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .status-badge-warning {
        background-color: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid #eab308;
        padding: 16px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .status-badge-danger {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 16px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 3. DIRECTORY SETUP, SESSION STATE & MODEL CACHING
# -------------------------------------------------------------
KNOWN_FACES_DIR = "known_faces"
EMBEDDINGS_PKL = "embeddings.pkl"
ATTENDANCE_CSV = "attendance.csv"

# Admin session state initialization
if "is_admin_authenticated" not in st.session_state:
    st.session_state["is_admin_authenticated"] = False

# Initializing CSV file structure
init_attendance_file(ATTENDANCE_CSV)

@st.cache_resource(show_spinner="⚡ Loading FaceNet (MTCNN + InceptionResnetV1) Models...")
def get_cached_models():
    """Models ko cache karta hai taake bar bar reload hone ka delay na aaye."""
    return load_facenet_models()

mtcnn, resnet, device = get_cached_models()

# Load/Sync embeddings cache
if "known_embeddings" not in st.session_state or st.session_state.get("reload_db", False):
    st.session_state["known_embeddings"] = load_or_update_embeddings(
        KNOWN_FACES_DIR, EMBEDDINGS_PKL, mtcnn, resnet, device
    )
    st.session_state["reload_db"] = False

known_embeddings = st.session_state["known_embeddings"]


# -------------------------------------------------------------
# 4. SIDEBAR CONTROLS & SYSTEM STATUS
# -------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ System Panel")
    st.markdown("---")
    
    # System Status badges
    st.markdown(f"**Compute Device:** `{device.type.upper()}`")
    st.markdown(f"**Registered Students:** `{len(known_embeddings)}`")
    st.markdown(f"**Model:** `InceptionResnetV1 (vggface2)`")
    st.markdown(f"**Detector:** `MTCNN`")
    
    st.markdown("---")
    st.subheader("🎯 Recognition Settings")
    threshold = st.slider(
        "Cosine Distance Threshold",
        min_value=0.2,
        max_value=0.8,
        value=0.5,
        step=0.05,
        help="Lower = Stricter match, Higher = More lenient match. Colab Default: 0.5"
    )
    
    st.markdown("---")
    st.subheader("🔐 Admin Access Control")
    
    if is_first_time_setup(ADMIN_CONFIG_FILE):
        st.info("⚠️ Admin Password setup nahi hua. Tab 'Register New Student' par ja kar setup karein.")
    elif st.session_state["is_admin_authenticated"]:
        st.markdown("<span style='color: #4ade80; font-weight: bold;'>🟢 Admin Session Active</span>", unsafe_allow_html=True)
        if st.button("🚪 Logout Admin", key="sidebar_logout_btn", use_container_width=True):
            st.session_state["is_admin_authenticated"] = False
            st.rerun()
    else:
        st.markdown("<span style='color: #facc15; font-weight: bold;'>🔒 Public Mode (Admin Locked)</span>", unsafe_allow_html=True)
        with st.expander("🔑 Quick Admin Login"):
            side_pwd = st.text_input("Admin Password:", type="password", key="side_pwd_input")
            if st.button("Login", key="side_login_btn", use_container_width=True):
                if verify_admin_password(side_pwd, ADMIN_CONFIG_FILE):
                    st.session_state["is_admin_authenticated"] = True
                    st.success("✅ Logged in!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect Password!")

    st.markdown("---")
    if st.button("🔄 Sync Database / Refresh Embeddings", use_container_width=True):
        st.session_state["reload_db"] = True
        st.rerun()

    st.markdown("---")
    st.caption("AI Face Recognition Attendance System v1.0")


# -------------------------------------------------------------
# 5. MAIN APPLICATION HEADER
# -------------------------------------------------------------
st.markdown("""
<div class="header-card">
    <div class="header-title">👤 Face Recognition Attendance System</div>
    <div class="header-subtitle">Powered by PyTorch, MTCNN & InceptionResnetV1 | Automatic Local Attendance Management</div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 6. TABS LAYOUT
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📸 Take Attendance",
    "👤 Register New Student",
    "📊 View Attendance Log"
])


# =============================================================
# TAB 1: TAKE ATTENDANCE (Webcam / Live Capture - PUBLIC ACCESS)
# =============================================================
with tab1:
    st.subheader("📸 Live Webcam Face Recognition")
    st.markdown("Apna chehra camera ke samne layein aur **Take Photo** click karein. System automatically attendance mark kar dega.")

    col_cam, col_result = st.columns([1.1, 1], gap="medium")

    with col_cam:
        # Streamlit Browser Webcam Input
        camera_photo = st.camera_input("Capture Face for Attendance", key="attendance_cam")
        
        # Backup File Upload option
        with st.expander("📁 Alternative: Upload Photo instead of Webcam"):
            uploaded_test_img = st.file_uploader(
                "Upload a test photo",
                type=["jpg", "jpeg", "png", "webp"],
                key="test_img_upload"
            )

    # Determine which image to process
    active_img = camera_photo or uploaded_test_img

    with col_result:
        st.subheader("🔍 Recognition Result")
        
        if active_img is not None:
            with st.spinner("Processing face & matching embeddings..."):
                match_name, distance, face_crop = recognize_face(
                    active_img,
                    known_embeddings,
                    mtcnn,
                    resnet,
                    device,
                    threshold=threshold
                )

            # Display cropped face preview if available
            if face_crop is not None:
                st.image(face_crop, caption="Detected Face Crop", width=160)

            # Check match results
            if match_name == "No face detected in image":
                st.markdown("""
                <div class="status-badge-danger">
                    ⚠️ Face Not Detected!<br>
                    <span style="font-size: 0.9rem; font-weight: normal;">Kripya chehra camera ke center mein layein aur lighting sahi rakhein.</span>
                </div>
                """, unsafe_allow_html=True)
                
            elif match_name == "Unknown Person - Not Registered":
                st.markdown(f"""
                <div class="status-badge-danger">
                    ❌ Unknown Person - Not Registered<br>
                    <span style="font-size: 0.9rem; font-weight: normal;">Distance Score: <b>{distance:.3f}</b> (Threshold: {threshold})</span><br>
                    <span style="font-size: 0.85rem; font-weight: normal;">Yeh shakhs database mein registered nahi hai. Admin Tab 'Register New Student' par ja kar add karein.</span>
                </div>
                """, unsafe_allow_html=True)

            elif match_name == "No registered students in database":
                st.info("ℹ️ Database mein filhal koi student registered nahi hai. Pehle student register karein.")

            else:
                # Registered student found! Mark attendance in CSV
                success, msg = mark_attendance(match_name, ATTENDANCE_CSV)
                
                if success:
                    st.markdown(f"""
                    <div class="status-badge-success">
                        ✅ {msg}<br>
                        <span style="font-size: 0.9rem; font-weight: normal;">Student Name: <b>{match_name}</b> | Distance Score: <b>{distance:.3f}</b></span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                    <div class="status-badge-warning">
                        ⚠️ {msg}<br>
                        <span style="font-size: 0.9rem; font-weight: normal;">Student Name: <b>{match_name}</b> | Distance Score: <b>{distance:.3f}</b></span>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.info("👈 Camera se photo capture karein ya photo upload karein taake attendance mark ho sake.")


# =============================================================
# TAB 2: REGISTER NEW STUDENT & MANAGE (ADMIN PROTECTED)
# =============================================================
with tab2:
    if is_first_time_setup(ADMIN_CONFIG_FILE):
        st.subheader("🔐 First-Time Admin Password Setup")
        st.markdown("Is device par pehli baar app run ho rahi hai. Admin Management & Student Registration features ke liye password setup karein.")

        col_setup1, _ = st.columns([1.2, 1], gap="medium")
        with col_setup1:
            setup_p1 = st.text_input("Create Admin Password:", type="password", key="setup_pwd_1")
            setup_p2 = st.text_input("Confirm Admin Password:", type="password", key="setup_pwd_2")
            btn_setup = st.button("✨ Set Admin Password & Save", type="primary", use_container_width=True)

            if btn_setup:
                s_ok, s_msg = setup_admin_password(setup_p1, setup_p2, ADMIN_CONFIG_FILE)
                if s_ok:
                    st.session_state["is_admin_authenticated"] = True
                    st.success("✅ Admin Password successfully created! You are now logged in as Admin.")
                    st.rerun()
                else:
                    st.error(f"❌ Setup Failed: {s_msg}")

    elif not st.session_state["is_admin_authenticated"]:
        st.subheader("🔒 Admin Login Required")
        st.markdown("Student Registration aur Deletion features access karne ke liye Admin Password enter karein.")

        col_l1, _ = st.columns([1.2, 1], gap="medium")
        with col_l1:
            tab_pwd = st.text_input("Enter Admin Password:", type="password", key="tab_pwd_input")
            tab_login_btn = st.button("🔑 Login as Admin", type="primary", use_container_width=True)

            if tab_login_btn:
                if verify_admin_password(tab_pwd, ADMIN_CONFIG_FILE):
                    st.session_state["is_admin_authenticated"] = True
                    st.success("✅ Login Successful! Welcome Admin.")
                    st.rerun()
                else:
                    st.error("❌ Incorrect Password! Access denied.")

    else:
        # Admin is Logged in! Show logout button header + full Registration & Management UI
        c_head1, c_head2 = st.columns([3, 1])
        with c_head1:
            st.markdown("<span style='color: #4ade80; font-weight: bold; font-size: 1.1rem;'>🟢 Admin Session Active</span>", unsafe_allow_html=True)
        with c_head2:
            if st.button("🚪 Logout Admin", key="tab2_logout_btn", use_container_width=True):
                st.session_state["is_admin_authenticated"] = False
                st.rerun()

        st.markdown("---")
        st.subheader("👤 Register New Student")
        st.markdown("Student ki reference photo upload/capture karein aur unka naam enter karein. System khud image ko save aur embedding generate kar le ga.")

        col_reg_input, col_reg_preview = st.columns([1.1, 1], gap="medium")

        with col_reg_input:
            student_name_input = st.text_input("Enter Student Full Name:", placeholder="e.g. Ali Zain", key="reg_name")
            
            reg_source = st.radio("Choose Photo Source:", ["Upload Photo File", "Capture via Webcam"], horizontal=True)
            
            reg_img_file = None
            if reg_source == "Upload Photo File":
                reg_img_file = st.file_uploader(
                    "Upload Student Photo (koi bhi filename chalegi):",
                    type=["jpg", "jpeg", "png", "webp", "bmp"],
                    key="reg_file_upload"
                )
            else:
                reg_img_file = st.camera_input("Capture Student Photo for Registration", key="reg_cam")

            submit_registration = st.button("✨ Save & Register Student", type="primary", use_container_width=True)

        with col_reg_preview:
            st.subheader("📋 Registration Status")
            
            if submit_registration:
                if not student_name_input or not student_name_input.strip():
                    st.error("⚠️ Please enter student name before registering!")
                elif reg_img_file is None:
                    st.error("⚠️ Please upload or capture a photo first!")
                else:
                    with st.spinner("Processing & embedding face..."):
                        success, msg, face_crop = register_student(
                            reg_img_file,
                            student_name_input,
                            KNOWN_FACES_DIR,
                            EMBEDDINGS_PKL,
                            mtcnn,
                            resnet,
                            device
                        )

                    if success:
                        st.success(msg)
                        if face_crop is not None:
                            st.image(face_crop, caption=f"Cropped Registered Face: {student_name_input}", width=160)
                        
                        # Update session state
                        st.session_state["reload_db"] = True
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.info("💡 Form fill karke 'Save & Register Student' click karein.")

        # -------------------------------------------------------------
        # MANAGE & DELETE REGISTERED STUDENTS SECTION
        # -------------------------------------------------------------
        st.markdown("---")
        st.subheader("🗑️ Manage / Delete Registered Students")
        st.markdown("Database se kisi student ko remove karne ke liye neeche dropdown se select karein aur **Delete Student** click karein.")

        student_list = sorted(list(known_embeddings.keys()))

        if student_list:
            col_del_select, col_del_action = st.columns([1.5, 1], gap="medium")

            with col_del_select:
                selected_student_to_delete = st.selectbox(
                    "Select Registered Student to Remove:",
                    options=student_list,
                    key="delete_student_select"
                )

            with col_del_action:
                st.markdown("<br>", unsafe_allow_html=True)
                delete_btn = st.button("🗑️ Delete Student", type="secondary", use_container_width=True)

            if delete_btn and selected_student_to_delete:
                with st.spinner(f"Deleting '{selected_student_to_delete}'..."):
                    del_success, del_msg = delete_student(
                        selected_student_to_delete,
                        KNOWN_FACES_DIR,
                        EMBEDDINGS_PKL
                    )

                if del_success:
                    st.success(f"✅ Student '{selected_student_to_delete}' successfully deleted from database and known_faces folder!")
                    st.session_state["reload_db"] = True
                    st.rerun()
                else:
                    st.error(f"❌ Deletion failed: {del_msg}")
        else:
            st.info("ℹ️ Abhi database mein koi student registered nahi hai.")


# =============================================================
# TAB 3: VIEW ATTENDANCE LOG & CSV EXPORT (PUBLIC ACCESS)
# =============================================================
with tab3:
    st.subheader("📊 Attendance History & Logs")
    
    # Reload records
    df_records, metrics = get_attendance_records(ATTENDANCE_CSV)

    # Metric Dashboard
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(known_embeddings)}</div>
            <div class="metric-label">Registered Students</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['total_entries']}</div>
            <div class="metric-label">Total Log Entries</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['today_count']}</div>
            <div class="metric-label">Today's Present Logs</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['unique_today']}</div>
            <div class="metric-label">Unique Students Today</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
    with f_col1:
        name_search = st.text_input("🔍 Search by Student Name:", placeholder="Type name...", key="filter_name")
    with f_col2:
        enable_date_filter = st.checkbox("Filter by Specific Date")
        selected_date = None
        if enable_date_filter:
            selected_date = st.date_input("Select Date:", key="filter_date")
    with f_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        # Refresh table button
        if st.button("🔄 Refresh Logs Table", use_container_width=True):
            st.rerun()

    # Get filtered dataset
    filtered_df, _ = get_attendance_records(ATTENDANCE_CSV, date_filter=selected_date, name_filter=name_search)

    st.markdown("### Attendance Records")
    if not filtered_df.empty:
        # Display data table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "Name": st.column_config.TextColumn("Student Name"),
                "Date": st.column_config.TextColumn("Date (YYYY-MM-DD)"),
                "Time": st.column_config.TextColumn("Time (HH:MM:SS)")
            },
            hide_index=True
        )

        # CSV Export button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Attendance CSV",
            data=csv_data,
            file_name="attendance_report.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.warning("No attendance records found matching the specified filters.")
