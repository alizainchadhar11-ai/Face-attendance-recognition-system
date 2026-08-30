import os
import sys
import pickle
import numpy as np
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1

# Reconfigure stdout for UTF-8 support on Windows default terminal (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# -------------------------------------------------------------
# 1. MODELS LOADING FUNCTION
# MTCNN face detection aur InceptionResnetV1 embeddings ke liye
# -------------------------------------------------------------
def load_facenet_models():
    """
    MTCNN face detector aur InceptionResnetV1 (vggface2 pretrained) model load karta hai.
    GPU agar available ho to CUDA use karega, warna CPU.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Face detector configuration (Matching original Colab notebook)
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        post_process=True,
        device=device
    )
    
    # Pretrained face recognition model (VGGFace2 dataset pe trained)
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
# Notebook logic: 1 - (dot_product / (norm_a * norm_b))
# -------------------------------------------------------------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Do face embeddings ke beech Cosine Distance calculate karta hai.
    0 ka matlab identical faces, > 0.5 ka matlab different persons.
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
# Image se face detect karke 512-d embedding vector nikalta hai
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
            # EXIF orientation fix for mobile/webcam photos
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

        # Cropped face ko display ke liye PIL image banayein
        face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
        face_np = (face_np - face_np.min()) / (face_np.max() - face_np.min() + 1e-5) * 255.0
        face_pil = Image.fromarray(face_np.astype(np.uint8))

        return embedding_np, face_pil, None

    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"


# -------------------------------------------------------------
# 4. LOAD & UPDATE EMBEDDINGS CACHE (embeddings.pkl)
# Pehle se saved embeddings read karta hai, aur sirf naye photos
# ki embedding compute karke pickle file update karta hai.
# -------------------------------------------------------------
def load_or_update_embeddings(known_faces_dir: str, pkl_path: str, mtcnn, resnet, device):
    """
    known_faces directory se photos scan karke embeddings.pkl load/update karta hai.
    Purani embeddings dobara generate nahi hoti.
    """
    os.makedirs(known_faces_dir, exist_ok=True)
    known_embeddings = {}

    # Read existing pickle cache if present
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                known_embeddings = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {pkl_path}, creating fresh cache. Error: {e}")
            known_embeddings = {}

    updated = False
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    # Find missing/new faces in known_faces_dir
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(valid_extensions):
            person_name = os.path.splitext(filename)[0]

            # Agar student ki embedding pehle se saved nahi hai to generate karo
            if person_name not in known_embeddings:
                filepath = os.path.join(known_faces_dir, filename)
                emb, _, err = get_embedding(filepath, mtcnn, resnet, device)
                if emb is not None:
                    known_embeddings[person_name] = emb
                    updated = True
                    print(f"[OK] Incremental embedding generated for: {person_name}")
                else:
                    print(f"[WARN] Face detection failed for reference file: {filename}")

    # Save to pickle if any new student was added
    if updated or not os.path.exists(pkl_path):
        with open(pkl_path, 'wb') as f:
            pickle.dump(known_embeddings, f)
        print(f"[SAVED] Updated embeddings saved to {pkl_path}")

    return known_embeddings


# -------------------------------------------------------------
# 5. REGISTER NEW STUDENT
# Nayi photo upload + Naam enter karne par auto-rename, save & embed
# -------------------------------------------------------------
def register_student(image_input, student_name: str, known_faces_dir: str, pkl_path: str, mtcnn, resnet, device):
    """
    Naye student ko register karta hai:
    1. Student name sanitize karta hai
    2. Image se face detect karta hai
    3. Image ko known_faces/<CleanName>.jpg ke naam se save karta hai
    4. Embedding generate karke embeddings.pkl update karta hai
    """
    # Name validation & sanitization
    clean_name = "".join([c for c in student_name.strip() if c.isalnum() or c in (" ", "_", "-")]).strip()
    if not clean_name:
        return False, "Khabardar: Student name invalid ya khali hai! Kripya sahi naam enter karein.", None

    # Step 1: Detect face & extract embedding
    emb, face_pil, err = get_embedding(image_input, mtcnn, resnet, device)
    if emb is None:
        return False, f"Registration Failed: {err or 'Face not detected in uploaded image!'}", None

    # Step 2: Save original reference image to known_faces folder
    os.makedirs(known_faces_dir, exist_ok=True)
    target_filename = f"{clean_name}.jpg"
    target_path = os.path.join(known_faces_dir, target_filename)

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

    # Step 3: Load existing embeddings, add new student, and save pickle
    known_embeddings = {}
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                known_embeddings = pickle.load(f)
        except Exception:
            known_embeddings = {}

    known_embeddings[clean_name] = emb

    with open(pkl_path, 'wb') as f:
        pickle.dump(known_embeddings, f)

    return True, f"[SUCCESS] Student '{clean_name}' successfully registered and saved to known_faces/{target_filename}!", face_pil


# -------------------------------------------------------------
# 6. FACE RECOGNITION FUNCTION
# Test image ko DB ke tamam known_embeddings se compare karta hai
# -------------------------------------------------------------
def recognize_face(image_input, known_embeddings: dict, mtcnn, resnet, device, threshold: float = 0.5):
    """
    Test image ko known_embeddings se compare karke best match return karta hai.
    Returns: (match_name, distance, face_crop)
    """
    if not known_embeddings:
        return "No registered students in database", None, None

    test_embedding, face_pil, err = get_embedding(image_input, mtcnn, resnet, device)

    if test_embedding is None:
        return "No face detected in image", None, None

    best_match = None
    best_distance = float('inf')

    # Cosine distance computation for all known faces
    for name, known_embedding in known_embeddings.items():
        dist = cosine_distance(test_embedding, known_embedding)
        if dist < best_distance:
            best_distance = dist
            best_match = name

    # Threshold check (Colab notebook threshold = 0.5)
    if best_distance < threshold:
        return best_match, best_distance, face_pil
    else:
        return "Unknown Person - Not Registered", best_distance, face_pil


# -------------------------------------------------------------
# 7. DELETE STUDENT FUNCTION
# Student ko embeddings.pkl aur known_faces folder dono se remove karta hai
# -------------------------------------------------------------
def delete_student(student_name: str, known_faces_dir: str, pkl_path: str):
    """
    Registered student ko embeddings.pkl cache aur known_faces folder se remove karta hai.
    Returns: (success_bool, message_str)
    """
    clean_name = student_name.strip()
    if not clean_name:
        return False, "Student name invalid hai."

    photo_deleted = False
    embedding_deleted = False

    # Step 1: Delete image from known_faces directory
    if os.path.exists(known_faces_dir):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for ext in valid_extensions:
            target_path = os.path.join(known_faces_dir, f"{clean_name}{ext}")
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                    photo_deleted = True
                    break
                except Exception as e:
                    return False, f"Failed to delete photo file: {str(e)}"

    # Step 2: Remove student entry from embeddings.pkl
    known_embeddings = {}
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                known_embeddings = pickle.load(f)
        except Exception as e:
            return False, f"Failed to load embeddings file: {str(e)}"

    if clean_name in known_embeddings:
        del known_embeddings[clean_name]
        embedding_deleted = True
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(known_embeddings, f)
        except Exception as e:
            return False, f"Failed to update embeddings file: {str(e)}"

    if photo_deleted or embedding_deleted:
        return True, f"[SUCCESS] Student '{clean_name}' successfully deleted from database and known_faces folder."
    else:
        return False, f"[WARN] Student '{clean_name}' not found in database or folder."

