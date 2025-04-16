import os
import hashlib
from dotenv import load_dotenv
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

dotenv_path = "../server/.env"
load_dotenv(dotenv_path)

def decrypt_api_key(encrypted_key):
    try:
        iv_hex, encrypted_hex = encrypted_key.split(":")
        iv = bytes.fromhex(iv_hex)
        ciphertext = bytes.fromhex(encrypted_hex)
        # Derive a 32-byte key using your secret (ensure ENCRYPTION_SECRET is set in your .env)
        secret = os.getenv("ENCRYPTION_SECRET").encode("utf-8")
        salt = b"salt"
        key = hashlib.scrypt(secret, salt=salt, n=16384, r=8, p=1, dklen=32)
        # Create a cipher for AES-256-CBC decryption.
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        # Remove PKCS#7 padding.
        padding_len = padded_plaintext[-1]
        plaintext = padded_plaintext[:-padding_len]
        return plaintext.decode("utf-8")
    except Exception as e:
        print("Decryption error:", e)
        raise
