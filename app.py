import os
import sys
import datetime
import streamlit as st
from PIL import Image
import pandas as pd

# Reconfigure stdout for UTF-8 support on Windows default terminal (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from utils.db_utils import (
    init_school_db,
    get_all_classes,
    add_class,
    delete_class,
    get_all_staff,
    delete_staff,
    get_all_students,
    get_students_by_class
)
from utils.auth_utils import (
    is_first_time_setup,
    setup_admin_password,
    verify_admin_password,
    create_staff_account,
    verify_staff_password,
    ADMIN_CONFIG_FILE
)
from utils.face_utils import (
    load_facenet_models,
    load_or_update_embeddings_classwise,
    register_student_classwise,
    delete_student_classwise,
    recognize_face_scoped
)
from utils.attendance_utils import (
    init_attendance_file,
    mark_attendance_scoped,
    get_attendance_records,
    ATTENDANCE_CSV_FILE
)
from utils.report_utils import (
    bulk_import_students_from_csv,
    get_student_attendance_summary,
    get_class_subject_analytics,
    get_absentee_list,
    generate_excel_report
)

# -------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="School Attendance Management System",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
# 2. GLOBAL SETUP & MODEL CACHING
# -------------------------------------------------------------
KNOWN_FACES_DIR = "known_faces"
EMBEDDINGS_PKL = "embeddings.pkl"

init_school_db()
init_attendance_file(ATTENDANCE_CSV_FILE)

# Session state initializations
if "user_role" not in st.session_state:
    st.session_state["user_role"] = None  # Options: None, "admin", "staff"

if "logged_user" not in st.session_state:
    st.session_state["logged_user"] = None

@st.cache_resource(show_spinner="⚡ Loading FaceNet (MTCNN + InceptionResnetV1) Models...")
def get_cached_models():
    return load_facenet_models()

mtcnn, resnet, device = get_cached_models()

# Sync classwise embeddings cache
if "known_embeddings" not in st.session_state or st.session_state.get("reload_db", False):
    st.session_state["known_embeddings"] = load_or_update_embeddings_classwise(
        KNOWN_FACES_DIR, EMBEDDINGS_PKL, mtcnn, resnet, device
    )
    st.session_state["reload_db"] = False

known_embeddings = st.session_state["known_embeddings"]
all_classes_dict = get_all_classes()


# -------------------------------------------------------------
# 3. SIDEBAR SYSTEM PANEL (DYNAMIC BASED ON ROLE)
# -------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ System Panel")
    st.markdown("---")
    
    current_role = st.session_state["user_role"]
    logged_user = st.session_state["logged_user"]

    if current_role == "admin":
        st.markdown("<span style='color: #4ade80; font-weight: bold; font-size: 1.1rem;'>🟢 Role: ADMIN</span>", unsafe_allow_html=True)
    elif current_role == "staff" and logged_user:
        st.markdown(f"<span style='color: #38bdf8; font-weight: bold; font-size: 1.1rem;'>👨‍🏫 Staff: {logged_user.get('name')}</span>", unsafe_allow_html=True)
        st.caption(f"Username: @{logged_user.get('username')}")
    else:
        st.markdown("<span style='color: #94a3b8; font-weight: bold;'>🔒 Not Logged In</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Compute Device:** `{device.type.upper()}`")
    st.markdown(f"**Total Classes:** `{len(all_classes_dict)}`")
    st.markdown(f"**Total Students:** `{len(get_all_students())}`")
    st.markdown(f"**Total Staff:** `{len(get_all_staff())}`")
    
    st.markdown("---")
    threshold = st.slider(
        "Cosine Distance Threshold",
        min_value=0.2,
        max_value=0.8,
        value=0.5,
        step=0.05,
        help="Colab Default: 0.5"
    )
    
    if current_role is not None:
        st.markdown("---")
        if st.button("🔄 Sync Database / Refresh", use_container_width=True):
            st.session_state["reload_db"] = True
            st.rerun()

        if st.button("🚪 Logout", type="primary", use_container_width=True):
            st.session_state["user_role"] = None
            st.session_state["logged_user"] = None
            st.rerun()

    st.markdown("---")
    st.caption("School Attendance Management System v2.0")


# -------------------------------------------------------------
# 4. MAIN HEADER BANNER
# -------------------------------------------------------------
st.markdown("""
<div class="header-card">
    <div class="header-title">🏫 School & Institute Management System</div>
    <div class="header-subtitle">AI-Powered Class-Wise Face Recognition Attendance Portal</div>
</div>
""", unsafe_allow_html=True)


# =============================================================
# VIEW 1: ROLE SELECTION & LOGIN SCREEN (WHEN NOT LOGGED IN)
# =============================================================
if st.session_state["user_role"] is None:
    st.subheader("🔑 Select Login Portal")
    
    login_tab_choice = st.radio(
        "Choose Portal:",
        ["Admin Portal", "Staff / Teacher Portal"],
        horizontal=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # ADMIN PORTAL LOGIN / FIRST-TIME SETUP
    # ---------------------------------------------------------
    if login_tab_choice == "Admin Portal":
        if is_first_time_setup(ADMIN_CONFIG_FILE):
            st.subheader("🔐 First-Time Admin Password Setup")
            st.info("App pehli baar setup ho rahi hai. Kripya Admin Dashboard ke liye naya password set karein.")
            
            col_s1, _ = st.columns([1.2, 1])
            with col_s1:
                p1 = st.text_input("Create Admin Password:", type="password", key="init_p1")
                p2 = st.text_input("Confirm Admin Password:", type="password", key="init_p2")
                btn_init = st.button("✨ Set Admin Password & Continue", type="primary", use_container_width=True)
                
                if btn_init:
                    ok, msg = setup_admin_password(p1, p2, ADMIN_CONFIG_FILE)
                    if ok:
                        st.session_state["user_role"] = "admin"
                        st.success("✅ Admin password created! Logged in as Admin.")
                        st.rerun()
                    else:
                        st.error(f"❌ Setup Failed: {msg}")
        else:
            st.subheader("🔒 Admin Login")
            col_a1, _ = st.columns([1.2, 1])
            with col_a1:
                admin_pwd = st.text_input("Enter Admin Password:", type="password", key="admin_login_pwd")
                btn_admin_login = st.button("🔑 Login as Admin", type="primary", use_container_width=True)
                
                if btn_admin_login:
                    if verify_admin_password(admin_pwd, ADMIN_CONFIG_FILE):
                        st.session_state["user_role"] = "admin"
                        st.success("✅ Welcome Admin!")
                        st.rerun()
                    else:
                        st.error("❌ Incorrect Admin Password!")

    # ---------------------------------------------------------
    # STAFF / TEACHER PORTAL LOGIN
    # ---------------------------------------------------------
    else:
        st.subheader("👨‍🏫 Staff / Teacher Login")
        st.markdown("Apna Username aur Password enter karke login karein.")
        
        col_st1, _ = st.columns([1.2, 1])
        with col_st1:
            staff_user_input = st.text_input("Enter Username:", placeholder="e.g. sir_ahmed", key="staff_user_in")
            staff_pwd_input = st.text_input("Enter Password:", type="password", key="staff_pwd_in")
            btn_staff_login = st.button("🔑 Login as Teacher", type="primary", use_container_width=True)
            
            if btn_staff_login:
                ok, msg, staff_data = verify_staff_password(staff_user_input, staff_pwd_input)
                if ok:
                    st.session_state["user_role"] = "staff"
                    st.session_state["logged_user"] = staff_data
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(f"❌ Login Failed: {msg}")


# =============================================================
# VIEW 2: ADMIN DASHBOARD (WHEN ROLE == "admin")
# =============================================================
elif st.session_state["user_role"] == "admin":
    st.markdown("### 🛠️ Admin Management Dashboard")
    
    admin_tab1, admin_tab2, admin_tab3, admin_tab4, admin_tab5 = st.tabs([
        "🏫 Manage Classes",
        "👨‍🏫 Manage Staff",
        "🎓 Manage Students",
        "📊 All Attendance Logs",
        "📈 Reports & Analytics"
    ])

    # ---------------------------------------------------------
    # TAB 1: MANAGE CLASSES
    # ---------------------------------------------------------
    with admin_tab1:
        st.subheader("🏫 Classes & Subjects Configuration")
        
        col_c_add, col_c_list = st.columns([1.2, 1], gap="medium")
        
        with col_c_add:
            st.markdown("#### ➕ Add New Class")
            new_class_name = st.text_input("Class Name:", placeholder="e.g. Class 9", key="add_c_name")
            new_subjects_str = st.text_area(
                "Subjects List (comma separated):",
                placeholder="e.g. Math, Physics, Chemistry, English",
                key="add_c_subs"
            )
            btn_add_class = st.button("✨ Save Class", type="primary", use_container_width=True)
            
            if btn_add_class:
                subs_list = [s.strip() for s in new_subjects_str.split(",") if s.strip()]
                c_ok, c_msg = add_class(new_class_name, subs_list)
                if c_ok:
                    st.success(c_msg)
                    st.session_state["reload_db"] = True
                    st.rerun()
                else:
                    st.error(c_msg)
                    
        with col_c_list:
            st.markdown("#### 📋 Existing Classes & Subjects")
            classes_dict = get_all_classes()
            if classes_dict:
                for c_name, s_list in classes_dict.items():
                    with st.expander(f"📚 {c_name} ({len(s_list)} Subjects)"):
                        st.write("**Subjects:**", ", ".join(s_list))
                        if st.button(f"🗑️ Delete {c_name}", key=f"del_c_{c_name}"):
                            del_ok, del_msg = delete_class(c_name)
                            if del_ok:
                                st.success(del_msg)
                                st.session_state["reload_db"] = True
                                st.rerun()
                            else:
                                st.error(del_msg)
            else:
                st.info("No classes added yet.")

    # ---------------------------------------------------------
    # TAB 2: MANAGE STAFF / TEACHERS
    # ---------------------------------------------------------
    with admin_tab2:
        st.subheader("👨‍🏫 Staff Accounts & Class Assignments")
        
        col_st_add, col_st_list = st.columns([1.2, 1], gap="medium")
        
        with col_st_add:
            st.markdown("#### ➕ Add New Teacher / Staff")
            st_name = st.text_input("Teacher Full Name:", placeholder="e.g. Sir Ahmed", key="st_name_in")
            st_user = st.text_input("Username (for login):", placeholder="e.g. sir_ahmed", key="st_user_in")
            st_pwd = st.text_input("Password:", type="password", key="st_pwd_in")
            
            st.markdown("##### Assign Classes & Subjects:")
            current_classes = get_all_classes()
            staff_assignments = {}
            
            if current_classes:
                for c_name, sub_list in current_classes.items():
                    with st.expander(f"Assign for {c_name}"):
                        assigned_subs = st.multiselect(
                            f"Select subjects for {c_name}:",
                            options=sub_list,
                            key=f"assign_{c_name}"
                        )
                        if assigned_subs:
                            staff_assignments[c_name] = assigned_subs
            else:
                st.warning("⚠️ Pehle Classes add karein taake staff ko assign kar sakein.")

            btn_add_staff = st.button("✨ Create Staff Account", type="primary", use_container_width=True)
            
            if btn_add_staff:
                if not st_name or not st_user or not st_pwd:
                    st.error("⚠️ Name, Username aur Password zaroori hain!")
                elif not staff_assignments:
                    st.error("⚠️ Kam se kam ek Class aur Subject assign karein!")
                else:
                    st_ok, st_msg = create_staff_account(st_name, st_user, st_pwd, staff_assignments)
                    if st_ok:
                        st.success(st_msg)
                        st.session_state["reload_db"] = True
                        st.rerun()
                    else:
                        st.error(st_msg)

        with col_st_list:
            st.markdown("#### 📋 Existing Staff Members")
            staff_members = get_all_staff()
            if staff_members:
                for u_name, s_info in staff_members.items():
                    with st.expander(f"👨‍🏫 {s_info.get('name')} (@{u_name})"):
                        st.write("**Assigned Classes & Subjects:**")
                        for cls, subs in s_info.get("assignments", {}).items():
                            st.write(f"- **{cls}:** {', '.join(subs)}")
                        if st.button(f"🗑️ Delete Staff @{u_name}", key=f"del_st_{u_name}"):
                            d_ok, d_msg = delete_staff(u_name)
                            if d_ok:
                                st.success(d_msg)
                                st.session_state["reload_db"] = True
                                st.rerun()
                            else:
                                st.error(d_msg)
            else:
                st.info("No staff accounts created yet.")

    # ---------------------------------------------------------
    # TAB 3: MANAGE STUDENTS & BULK IMPORT
    # ---------------------------------------------------------
    with admin_tab3:
        st.subheader("🎓 Student Registration & Management")
        
        # BULK IMPORT EXPANDER SECTION
        with st.expander("📥 Bulk Student Import (CSV Upload)", expanded=False):
            st.markdown("CSV file upload karke ek sath multiple students add karein. Required columns: **`Name`**, **`Roll_Number`**, **`Class`**")
            
            sample_csv_data = "Name,Roll_Number,Class\nAli Zain,101,Class 9\nSara Khan,102,Class 9\nAhmed Raza,201,Class 10\n"
            st.download_button(
                "📄 Download Sample CSV Template",
                data=sample_csv_data,
                file_name="sample_students_import.csv",
                mime="text/csv"
            )
            
            bulk_csv_file = st.file_uploader("Choose CSV File for Import:", type=["csv"], key="bulk_student_csv")
            if bulk_csv_file is not None:
                if st.button("✨ Import Students from CSV", type="primary", key="btn_bulk_import"):
                    with st.spinner("Processing CSV import..."):
                        imp_count, skip_count, imp_msg = bulk_import_students_from_csv(bulk_csv_file, KNOWN_FACES_DIR)
                    if imp_count > 0:
                        st.success(imp_msg)
                        st.session_state["reload_db"] = True
                        st.rerun()
                    else:
                        st.warning(imp_msg)

        st.markdown("---")
        classes_dict = get_all_classes()
        if not classes_dict:
            st.warning("⚠️ Student register karne se pehle 'Manage Classes' tab mein Class add karein.")
        else:
            col_s_reg, col_s_list = st.columns([1.1, 1], gap="medium")
            
            with col_s_reg:
                st.markdown("#### ➕ Single Student Registration")
                selected_class_for_reg = st.selectbox("Select Class:", options=list(classes_dict.keys()), key="reg_cls_select")
                s_name_in = st.text_input("Student Name:", placeholder="e.g. Ali Zain", key="reg_s_name")
                s_roll_in = st.text_input("Roll Number:", placeholder="e.g. 101", key="reg_s_roll")
                
                s_photo_source = st.radio("Photo Source:", ["Upload File", "Webcam Capture"], horizontal=True, key="reg_source_r")
                s_photo_file = None
                if s_photo_source == "Upload File":
                    s_photo_file = st.file_uploader("Upload Photo:", type=["jpg", "png", "jpeg", "webp"], key="s_file_up")
                else:
                    s_photo_file = st.camera_input("Capture Photo:", key="s_cam_cap")
                    
                btn_reg_student = st.button("✨ Save & Register Student", type="primary", use_container_width=True)
                
                if btn_reg_student:
                    if not s_name_in or not s_roll_in:
                        st.error("⚠️ Student Name aur Roll Number dono zaroori hain!")
                    elif s_photo_file is None:
                        st.error("⚠️ Student ki photo upload ya capture karein!")
                    else:
                        with st.spinner("Embedding & saving student face..."):
                            reg_ok, reg_msg, f_crop = register_student_classwise(
                                s_photo_file,
                                s_name_in,
                                s_roll_in,
                                selected_class_for_reg,
                                KNOWN_FACES_DIR,
                                EMBEDDINGS_PKL,
                                mtcnn,
                                resnet,
                                device
                            )
                        if reg_ok:
                            st.success(reg_msg)
                            if f_crop is not None:
                                st.image(f_crop, caption=f"Registered Face: {s_name_in}", width=150)
                            st.session_state["reload_db"] = True
                            st.rerun()
                        else:
                            st.error(reg_msg)

            with col_s_list:
                st.markdown("#### 📋 View & Delete Students")
                selected_class_for_view = st.selectbox("Filter Students by Class:", options=list(classes_dict.keys()), key="view_cls_select")
                
                class_students = get_students_by_class(selected_class_for_view)
                if class_students:
                    st.write(f"Total Students in **{selected_class_for_view}**: `{len(class_students)}`")
                    for s_key, s_data in class_students.items():
                        c_col1, c_col2 = st.columns([3, 1])
                        with c_col1:
                            st.write(f"👤 **{s_data.get('name')}** (Roll: `{s_data.get('roll_number')}`) ")
                        with c_col2:
                            if st.button("🗑️ Delete", key=f"del_stud_{s_key}"):
                                s_del_ok, s_del_msg = delete_student_classwise(
                                    selected_class_for_view,
                                    s_data.get('roll_number'),
                                    KNOWN_FACES_DIR,
                                    EMBEDDINGS_PKL
                                )
                                if s_del_ok:
                                    st.success(s_del_msg)
                                    st.session_state["reload_db"] = True
                                    st.rerun()
                                else:
                                    st.error(s_del_msg)
                else:
                    st.info(f"No students registered in {selected_class_for_view}.")

    # ---------------------------------------------------------
    # TAB 4: ALL ATTENDANCE LOGS
    # ---------------------------------------------------------
    with admin_tab4:
        st.subheader("📊 Global Attendance Records")
        
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            classes_option = ["All Classes"] + list(get_all_classes().keys())
            c_filter = st.selectbox("Class Filter:", options=classes_option, key="adm_c_filt")
        with f_col2:
            sub_options = ["All Subjects"]
            if c_filter != "All Classes" and c_filter in get_all_classes():
                sub_options += get_all_classes()[c_filter]
            s_filter = st.selectbox("Subject Filter:", options=sub_options, key="adm_s_filt")
        with f_col3:
            enable_d = st.checkbox("Filter Date", key="adm_enable_d")
            d_filter = st.date_input("Select Date:", key="adm_d_filt") if enable_d else None
        with f_col4:
            search_query = st.text_input("Search Name/Roll:", placeholder="Type name or roll...", key="adm_q_filt")

        df_logs, metrics = get_attendance_records(
            ATTENDANCE_CSV_FILE,
            date_filter=d_filter,
            class_filter=c_filter,
            subject_filter=s_filter,
            name_filter=search_query
        )

        st.markdown("### Records Table")
        if not df_logs.empty:
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
            csv_bytes = df_logs.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Filtered Attendance CSV",
                data=csv_bytes,
                file_name="school_attendance_report.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.info("ℹ️ Abhi tak koi attendance record nahi hai.")

    # ---------------------------------------------------------
    # TAB 5: REPORTS & ANALYTICS (PHASE 2 FEATURE)
    # ---------------------------------------------------------
    with admin_tab5:
        st.subheader("📈 Attendance Reports, Analytics & Exports")
        
        # Date Filter Controls
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            preset_choice = st.selectbox(
                "🗓️ Date Range Preset:",
                options=["All Time", "Today", "Last 7 Days", "This Month", "Custom Date Range"],
                key="rep_preset"
            )
        
        start_d, end_d = None, None
        today_dt = datetime.date.today()
        if preset_choice == "Today":
            start_d, end_d = today_dt, today_dt
        elif preset_choice == "Last 7 Days":
            start_d, end_d = today_dt - datetime.timedelta(days=7), today_dt
        elif preset_choice == "This Month":
            start_d, end_d = today_dt.replace(day=1), today_dt
        elif preset_choice == "Custom Date Range":
            with r_col2:
                start_d = st.date_input("Start Date:", value=today_dt - datetime.timedelta(days=30), key="rep_start")
            with r_col3:
                end_d = st.date_input("End Date:", value=today_dt, key="rep_end")

        with r_col4:
            rep_class_filter = st.selectbox("Class Filter:", options=["All Classes"] + list(get_all_classes().keys()), key="rep_cls_f")

        st.markdown("---")

        # Fetch Analytics Data
        summary_df = get_student_attendance_summary(start_d, end_d, class_filter=rep_class_filter)
        class_chart_df, subject_chart_df = get_class_subject_analytics(start_d, end_d, class_filter=rep_class_filter)

        # Overview Metrics Banner
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            avg_pct = round(summary_df["Attendance_%"].mean(), 1) if not summary_df.empty else 0.0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_pct}%</div>
                <div class="metric-label">Overall Avg Attendance</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            tot_students = len(summary_df) if not summary_df.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{tot_students}</div>
                <div class="metric-label">Total Students Tracked</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            p_good = len(summary_df[summary_df["Attendance_%"] >= 75]) if not summary_df.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #4ade80;">{p_good}</div>
                <div class="metric-label">Good Standing (≥ 75%)</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            p_low = len(summary_df[summary_df["Attendance_%"] < 50]) if not summary_df.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #f87171;">{p_low}</div>
                <div class="metric-label">Low Attendance (< 50%)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Charts Section
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### 📊 Class-Wise Attendance %")
            if not class_chart_df.empty:
                st.bar_chart(class_chart_df.set_index("Class")["Attendance_%"])
            else:
                st.info("No class percentage data available.")

        with chart_col2:
            st.markdown("#### 📚 Subject Attendance Activity")
            if not subject_chart_df.empty:
                st.bar_chart(subject_chart_df.set_index("Subject")["Total_Present_Logs"])
            else:
                st.info("No subject activity data available.")

        st.markdown("---")
        st.markdown("### 🎓 Student Attendance Percentage Summary")
        if not summary_df.empty:
            # Color-coded badge formatting
            def format_badge(val):
                if val >= 75:
                    return f"🟢 {val}%"
                elif val >= 50:
                    return f"🟡 {val}%"
                else:
                    return f"🔴 {val}%"

            disp_df = summary_df.copy()
            disp_df["Status"] = disp_df["Attendance_%"].apply(format_badge)
            
            st.dataframe(
                disp_df[["Roll_Number", "Name", "Class", "Present_Days", "Absent_Days", "Total_Sessions", "Attendance_%", "Status"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No student summary data available.")

        # DAILY ABSENTEE TRACKER SECTION
        st.markdown("---")
        st.markdown("### ❌ Daily Absentee Tracker")
        st.markdown("Select date and class to view students who were **ABSENT**.")

        ab_c1, ab_c2 = st.columns(2)
        with ab_c1:
            absent_date = st.date_input("Target Date for Absentees:", value=today_dt, key="abs_date")
        with ab_c2:
            absent_cls = st.selectbox("Select Class:", options=list(get_all_classes().keys()) if get_all_classes() else ["None"], key="abs_cls")

        if get_all_classes():
            absentee_df = get_absentee_list(absent_date, class_filter=absent_cls)
            if not absentee_df.empty:
                st.dataframe(absentee_df, use_container_width=True, hide_index=True)
            else:
                st.success(f"🎉 Great news! 100% attendance in {absent_cls} on {absent_date} (No absentees).")

        # REPORT EXPORT SECTION
        st.markdown("---")
        st.markdown("### 📥 Download Reports (Excel & CSV)")
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            # Excel Report Generation
            raw_logs_df, _ = get_attendance_records(ATTENDANCE_CSV_FILE)
            abs_df_all = get_absentee_list(today_dt, class_filter=rep_class_filter)
            excel_bytes = generate_excel_report(summary_df, abs_df_all, raw_logs_df)
            
            st.download_button(
                "📊 Export Complete Report (Excel .xlsx)",
                data=excel_bytes,
                file_name=f"School_Attendance_Report_{today_dt}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )

        with exp_col2:
            csv_rep_bytes = summary_df.to_csv(index=False).encode('utf-8') if not summary_df.empty else b""
            st.download_button(
                "📄 Download Student Summary (CSV)",
                data=csv_rep_bytes,
                file_name=f"Student_Attendance_Summary_{today_dt}.csv",
                mime="text/csv",
                use_container_width=True
            )


# =============================================================
# VIEW 3: STAFF / TEACHER DASHBOARD (WHEN ROLE == "staff")
# =============================================================
elif st.session_state["user_role"] == "staff":
    staff_user = st.session_state["logged_user"]
    teacher_name = staff_user.get("name", "Teacher")
    assignments = staff_user.get("assignments", {})

    st.subheader(f"👨‍🏫 Welcome, {teacher_name}!")
    st.markdown("Apni Class aur Subject select karke students ki face attendance mark karein.")

    if not assignments:
        st.warning("⚠️ Aapko filhal koi Class ya Subject assign nahi kiya gaya. Kripya Admin se rabta karein.")
    else:
        # Step 1: Select Assigned Class
        assigned_classes_list = list(assignments.keys())
        selected_class = st.selectbox(
            "📍 Step 1: Select Class for Attendance:",
            options=assigned_classes_list,
            key="teacher_class_select"
        )

        # Step 2: Select Assigned Subject for the selected class
        assigned_subjects = assignments.get(selected_class, [])
        selected_subject = st.selectbox(
            f"📚 Step 2: Select Subject for {selected_class}:",
            options=assigned_subjects,
            key="teacher_subject_select"
        )

        st.markdown("---")
        st.subheader(f"📸 Live Attendance: {selected_class} - {selected_subject}")
        
        t_col_cam, t_col_res = st.columns([1.1, 1], gap="medium")

        with t_col_cam:
            cam_photo = st.camera_input(f"Capture Face for {selected_class} ({selected_subject})", key="teacher_cam")
            with st.expander("📁 Alternative: Upload Test Photo"):
                up_photo = st.file_uploader("Upload Image:", type=["jpg", "png", "jpeg", "webp"], key="teacher_up")

        t_active_img = cam_photo or up_photo

        with t_col_res:
            st.subheader("🔍 Recognition Result")
            if t_active_img is not None:
                with st.spinner(f"Matching face against {selected_class} students..."):
                    match_result, dist, f_crop = recognize_face_scoped(
                        t_active_img,
                        known_embeddings,
                        selected_class,
                        mtcnn,
                        resnet,
                        device,
                        threshold=threshold
                    )

                if f_crop is not None:
                    st.image(f_crop, caption="Detected Face Crop", width=150)

                if isinstance(match_result, dict):
                    # Match found!
                    s_name = match_result.get("name")
                    s_roll = match_result.get("roll_number")
                    
                    att_ok, att_msg = mark_attendance_scoped(
                        s_name,
                        s_roll,
                        selected_class,
                        selected_subject,
                        teacher_name,
                        ATTENDANCE_CSV_FILE
                    )

                    if att_ok:
                        st.markdown(f"""
                        <div class="status-badge-success">
                            ✅ {att_msg}<br>
                            <span style="font-size: 0.9rem; font-weight: normal;">Student: <b>{s_name}</b> (Roll: <b>{s_roll}</b>) | Match Score: <b>{dist:.3f}</b></span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f"""
                        <div class="status-badge-warning">
                            ⚠️ {att_msg}<br>
                            <span style="font-size: 0.9rem; font-weight: normal;">Student: <b>{s_name}</b> (Roll: <b>{s_roll}</b>) | Match Score: <b>{dist:.3f}</b></span>
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    # Error or Unknown person
                    st.markdown(f"""
                    <div class="status-badge-danger">
                        ❌ {match_result}<br>
                        <span style="font-size: 0.9rem; font-weight: normal;">Match Score: <b>{dist:.3f}</b> (Threshold: {threshold})</span>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.info("👈 Camera se photo capture karein ya image upload karein.")
