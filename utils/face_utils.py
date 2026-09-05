import os
import sys
import pickle
import numpy as np
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils.db_utils import add_student_record, delete_student_record

# Reconfigure stdout for UTF-8 support on Windows default terminal (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# -------------------------------------------------------------
# 1. MODELS LOADING FUNCTION
# -------------------------------------------------------------
def load_facenet_models():
    """
    MTCNN face detector aur InceptionResnetV1 (vggface2 pretrained) model load karta hai.
    GPU agar available ho to CUDA use karega, warna CPU.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        post_process=True,
        device=device
    )
    
    try:
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    except Exception as e:
        err_msg = str(e).lower()
        if "getaddrinfo" in err_msg or "url" in err_msg or "connection" in err_msg or "download" in err_msg:
            raise RuntimeError(
                "[ERROR] Network Error: InceptionResnetV1 model weights (vggface2) pehli baar download karne ke liye active Internet connection zaroori hai. "
                "Pehli baar internet connect karke app run karein taake weights cache (~/.cache/torch/checkpoints/) mein save ho jayein."
            ) from e
        raise e

    return mtcnn, resnet, device


# -------------------------------------------------------------
# 2. COSINE DISTANCE FUNCTION
# -------------------------------------------------------------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Do face embeddings ke beech Cosine Distance calculate karta hai.
    """
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 1.0
        
    return float(1.0 - (np.dot(a, b) / (norm_a * norm_b)))


# -------------------------------------------------------------
# 3. EMBEDDING GENERATION FUNCTION
# -------------------------------------------------------------
def get_embedding(image_input, mtcnn, resnet, device):
    """
    Image (filepath, BytesIO, ya PIL Image) se face crop aur embedding tensor return karta hai.
    Returns: (embedding_np, cropped_face_pil, error_message)
    """
    try:
        if hasattr(image_input, "seek"):
            image_input.seek(0)

        if isinstance(image_input, (str, os.PathLike)):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            img = Image.open(image_input).convert('RGB')
            img = ImageOps.exif_transpose(img)

        # Detect face & crop using MTCNN
        face_tensor = mtcnn(img)

        if face_tensor is None:
            return None, None, "Face not detected in image"

        # Generate face embedding tensor
        face_batch = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face_batch)

        embedding_np = embedding.cpu().numpy()

        # Cropped face for UI display
        face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
        face_np = (face_np - face_np.min()) / (face_np.max() - face_np.min() + 1e-5) * 255.0
        face_pil = Image.fromarray(face_np.astype(np.uint8))

        return embedding_np, face_pil, None

    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"


# -------------------------------------------------------------
# 4. LOAD & UPDATE EMBEDDINGS CACHE (CLASS-WISE)
# embeddings.pkl structure:
# {
#    "Class 9": {
#        "101": {"name": "Ali Zain", "roll_number": "101", "embedding": numpy_array}
#    }
# }
# -------------------------------------------------------------
def load_or_update_embeddings_classwise(known_faces_dir: str, pkl_path: str, mtcnn, resnet, device):
    """
    known_faces directory se Class-wise subfolders scan karke embeddings.pkl load/update karta hai.
    """
    os.makedirs(known_faces_dir, exist_ok=True)
    known_embeddings = {}

    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                known_embeddings = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {pkl_path}, creating fresh cache. Error: {e}")
            known_embeddings = {}

    updated = False
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    # Scan subdirectories in known_faces_dir (Each subfolder = Class Name)
    for item in os.listdir(known_faces_dir):
        class_folder = os.path.join(known_faces_dir, item)
        if os.path.isdir(class_folder):
            class_name = item  # e.g., "Class 9"
            if class_name not in known_embeddings:
                known_embeddings[class_name] = {}

            for filename in os.listdir(class_folder):
                if filename.lower().endswith(valid_extensions):
                    # Expected filename format: "<RollNumber>_<Name>.jpg" or "<Name>_<RollNumber>.jpg"
                    base_name = os.path.splitext(filename)[0]
                    parts = base_name.split('_')
                    if len(parts) >= 2:
                        roll_no = parts[0]
                        student_name = " ".join(parts[1:])
                    else:
                        roll_no = base_name
                        student_name = base_name

                    if roll_no not in known_embeddings[class_name]:
                        filepath = os.path.join(class_folder, filename)
                        emb, _, err = get_embedding(filepath, mtcnn, resnet, device)
                        if emb is not None:
                            known_embeddings[class_name][roll_no] = {
                                "name": student_name,
                                "roll_number": roll_no,
                                "embedding": emb
                            }
                            updated = True
                            print(f"[OK] Incremental class embedding generated: {class_name} - {student_name} ({roll_no})")
                        else:
                            print(f"[WARN] Face detection failed for reference file: {filepath}")

    if updated or not os.path.exists(pkl_path):
        with open(pkl_path, 'wb') as f:
            pickle.dump(known_embeddings, f)
        print(f"[SAVED] Class-wise embeddings saved to {pkl_path}")

    return known_embeddings


# -------------------------------------------------------------
# 5. REGISTER NEW STUDENT (CLASS-WISE)
# -------------------------------------------------------------
def register_student_classwise(image_input, student_name: str, roll_number: str, class_name: str, known_faces_dir: str, pkl_path: str, mtcnn, resnet, device):
    """
    Naye student ko class-wise register karta hai:
    1. Input sanitization
    2. Face detection check
    3. Save photo to known_faces/<Class_Name>/<Roll>_<Name>.jpg
    4. Save to school_db.json & update embeddings.pkl
    """
    clean_name = "".join([c for c in student_name.strip() if c.isalnum() or c in (" ", "_", "-")]).strip()
    clean_roll = "".join([c for c in roll_number.strip() if c.isalnum() or c in ("_", "-")]).strip()
    clean_class = class_name.strip()

    if not clean_name or not clean_roll or not clean_class:
        return False, "Student Name, Roll Number aur Class sab zaroori hain.", None

    # Step 1: Detect face & extract embedding
    emb, face_pil, err = get_embedding(image_input, mtcnn, resnet, device)
    if emb is None:
        return False, f"Registration Failed: {err or 'Uploaded image mein face detect nahi hua!'}", None

    # Step 2: Create class folder & save reference image
    class_folder = os.path.join(known_faces_dir, clean_class)
    os.makedirs(class_folder, exist_ok=True)
    
    target_filename = f"{clean_roll}_{clean_name.replace(' ', '')}.jpg"
    target_path = os.path.join(class_folder, target_filename)

    try:
        if hasattr(image_input, "seek"):
            image_input.seek(0)

        if isinstance(image_input, (str, os.PathLike)):
            img = Image.open(image_input).convert('RGB')
        else:
            img = Image.open(image_input).convert('RGB')
            img = ImageOps.exif_transpose(img)

        img.save(target_path, "JPEG", quality=95)
    except Exception as e:
        return False, f"Failed to save image file: {str(e)}", None

    # Step 3: Update school_db.json
    db_ok, db_msg = add_student_record(clean_name, clean_roll, clean_class, target_path)
    if not db_ok:
        return False, f"Failed to update database: {db_msg}", None

    # Step 4: Update embeddings.pkl
    known_embeddings = {}
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                known_embeddings = pickle.load(f)
        except Exception:
            known_embeddings = {}

    if clean_class not in known_embeddings:
        known_embeddings[clean_class] = {}

    known_embeddings[clean_class][clean_roll] = {
        "name": clean_name,
        "roll_number": clean_roll,
        "embedding": emb
    }

    with open(pkl_path, 'wb') as f:
        pickle.dump(known_embeddings, f)

    return True, f"[SUCCESS] Student '{clean_name}' (Roll: {clean_roll}) registered in {clean_class}!", face_pil


# -------------------------------------------------------------
# 6. DELETE STUDENT (CLASS-WISE)
# -------------------------------------------------------------
def delete_student_classwise(class_name: str, roll_number: str, known_faces_dir: str, pkl_path: str):
    """
    Student ko known_faces/<Class_Name>/, embeddings.pkl, aur school_db.json se delete karta hai.
    """
    clean_class = class_name.strip()
    clean_roll = roll_number.strip()

    if not clean_class or not clean_roll:
        return False, "Class Name aur Roll Number zaroori hain."

    # Step 1: Delete record from school_db.json
    delete_student_record(clean_class, clean_roll)

    # Step 2: Delete photo file from known_faces/<Class_Name>/
    class_folder = os.path.join(known_faces_dir, clean_class)
    photo_deleted = False
    if os.path.exists(class_folder):
        for fname in os.listdir(class_folder):
            if fname.startswith(f"{clean_roll}_") or os.path.splitext(fname)[0] == clean_roll:
                try:
                    os.remove(os.path.join(class_folder, fname))
                    photo_deleted = True
                except Exception as e:
                    print(f"[WARN] Could not remove photo file: {e}")

    # Step 3: Remove from embeddings.pkl
    known_embeddings = {}
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                known_embeddings = pickle.load(f)
        except Exception:
            known_embeddings = {}

    if clean_class in known_embeddings and clean_roll in known_embeddings[clean_class]:
        del known_embeddings[clean_class][clean_roll]
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(known_embeddings, f)
        except Exception as e:
            return False, f"Failed to save embeddings pickle: {str(e)}"

    return True, f"[SUCCESS] Student Roll #{clean_roll} deleted from {clean_class}."


# -------------------------------------------------------------
# 7. SCOPED FACE RECOGNITION FUNCTION (CLASS-SPECIFIC)
# Match ONLY against students belonging to class_name!
# -------------------------------------------------------------
def recognize_face_scoped(image_input, known_embeddings: dict, class_name: str, mtcnn, resnet, device, threshold: float = 0.5):
    """
    Test image ko SIRF class_name ke registered students se compare karta hai.
    Returns: (matched_student_dict_or_str, distance, face_crop)
    """
    class_embeddings = known_embeddings.get(class_name, {})
    if not class_embeddings:
        return f"No registered students found in {class_name}", None, None

    test_embedding, face_pil, err = get_embedding(image_input, mtcnn, resnet, device)
    if test_embedding is None:
        return "No face detected in image", None, None

    best_match_info = None
    best_distance = float('inf')

    # Cosine distance comparison ONLY for the specified class
    for roll_no, student_info in class_embeddings.items():
        known_emb = student_info.get("embedding")
        if known_emb is not None:
            dist = cosine_distance(test_embedding, known_emb)
            if dist < best_distance:
                best_distance = dist
                best_match_info = student_info

    # Threshold evaluation (default = 0.5)
    if best_distance < threshold and best_match_info is not None:
        return best_match_info, best_distance, face_pil
    else:
        return "Unknown Person - Not Registered in this Class", best_distance, face_pil
