import os
import sys
import json
import hashlib

# Reconfigure stdout for UTF-8 support on Windows default terminal (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ADMIN_CONFIG_FILE = "admin_config.json"


def _hash_password(password: str, salt: bytes = None) -> tuple[str, str]:
    """
    Password ko hashlib.pbkdf2_hmac (SHA-256 + Salt, 100,000 iterations) se hash karta hai.
    Returns: (hash_hex, salt_hex)
    """
    if salt is None:
        salt = os.urandom(16)
    
    hash_obj = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    return hash_obj.hex(), salt.hex()


def is_first_time_setup(filepath: str = ADMIN_CONFIG_FILE) -> bool:
    """
    Check karta hai ke kya pehli baar setup ho raha hai (admin_config.json exist nahi karti).
    """
    return not os.path.exists(filepath)


def setup_admin_password(password: str, confirm_password: str, filepath: str = ADMIN_CONFIG_FILE):
    """
    Pehli baar Admin password set aur save karta hai.
    Returns: (success_bool, message_str)
    """
    if not password or not password.strip():
        return False, "Password khali nahi ho sakta."
    
    if len(password) < 4:
        return False, "Password kam se kam 4 characters ka hona chahiye."
        
    if password != confirm_password:
        return False, "Passwords match nahi kar rahe. Kripya dobara check karein."
        
    try:
        hash_hex, salt_hex = _hash_password(password.strip())
        config_data = {
            "password_hash": hash_hex,
            "salt": salt_hex
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
            
        return True, "[SUCCESS] Admin password successfully created and saved!"
    except Exception as e:
        return False, f"Failed to save admin password: {str(e)}"


def verify_admin_password(password_input: str, filepath: str = ADMIN_CONFIG_FILE) -> bool:
    """
    Entered password ko saved hash se compare karke verify karta hai.
    """
    if not password_input or not os.path.exists(filepath):
        return False
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        stored_hash = config_data.get("password_hash")
        salt_hex = config_data.get("salt", "")
        if not stored_hash or not salt_hex:
            return False
            
        salt_bytes = bytes.fromhex(salt_hex)
        input_hash, _ = _hash_password(password_input.strip(), salt_bytes)
        return input_hash == stored_hash
    except Exception as e:
        print(f"[WARN] Failed to verify password: {e}")
        return False


def reset_admin_config(filepath: str = ADMIN_CONFIG_FILE) -> bool:
    """
    admin_config.json delete karke password reset ke liye system tayar karta hai.
    """
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return True
        except Exception:
            return False
    return True
