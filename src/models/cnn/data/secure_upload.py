import os
from cryptography.fernet import Fernet
from typing import Tuple, Optional
import logging
from pathlib import Path

class SecureFileHandler:
    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize the secure file handler with encryption capabilities.
        
        Args:
            encryption_key: Optional encryption key. If not provided, a new one will be generated.
        """
        self.logger = logging.getLogger(__name__)
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
    def encrypt_file(self, file_path: str) -> Tuple[bytes, str]:
        """
        Encrypt a file and return the encrypted data and filename.
        
        Args:
            file_path: Path to the file to encrypt
            
        Returns:
            Tuple of (encrypted_data, encrypted_filename)
        """
        try:
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            encrypted_data = self.cipher_suite.encrypt(file_data)
            encrypted_filename = f"{Path(file_path).stem}_encrypted{Path(file_path).suffix}"
            
            return encrypted_data, encrypted_filename
            
        except Exception as e:
            self.logger.error(f"Error encrypting file {file_path}: {str(e)}")
            raise
            
    def decrypt_file(self, encrypted_data: bytes, output_path: str) -> str:
        """
        Decrypt encrypted data and save it to the specified path.
        
        Args:
            encrypted_data: The encrypted data to decrypt
            output_path: Path where the decrypted file should be saved
            
        Returns:
            Path to the decrypted file
        """
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as file:
                file.write(decrypted_data)
                
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error decrypting file: {str(e)}")
            raise
            
    def save_encryption_key(self, key_path: str) -> None:
        """
        Save the encryption key to a file.
        
        Args:
            key_path: Path where the encryption key should be saved
        """
        try:
            with open(key_path, 'wb') as key_file:
                key_file.write(self.encryption_key)
        except Exception as e:
            self.logger.error(f"Error saving encryption key: {str(e)}")
            raise
            
    @classmethod
    def load_encryption_key(cls, key_path: str) -> 'SecureFileHandler':
        """
        Load an encryption key from a file and create a new SecureFileHandler instance.
        
        Args:
            key_path: Path to the encryption key file
            
        Returns:
            New SecureFileHandler instance with the loaded key
        """
        try:
            with open(key_path, 'rb') as key_file:
                encryption_key = key_file.read()
            return cls(encryption_key)
        except Exception as e:
            logging.error(f"Error loading encryption key: {str(e)}")
            raise 