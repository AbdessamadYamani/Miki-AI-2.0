import keyring # <-- Add this import
import logging # type: ignore
import sys,os
from typing import Optional
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import AGENT_CREDENTIAL_SERVICE






def save_credential(service_name: str, username: str, secret: str):
    """Saves a credential securely using the OS keyring."""
    try:
        keyring.set_password(f"{AGENT_CREDENTIAL_SERVICE}_{service_name}", username, secret)
        logging.info(f"Successfully stored credential for service '{service_name}', user '{username}'.")
        return True
    except keyring.errors.KeyringError as e:
        logging.error(f"Failed to store credential for service '{service_name}', user '{username}': {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error storing credential: {e}", exc_info=True)
        return False

def get_credential(service_name: str, username: str) -> Optional[str]:
    """Retrieves a credential securely from the OS keyring."""
    try:
        secret = keyring.get_password(f"{AGENT_CREDENTIAL_SERVICE}_{service_name}", username)
        if secret:
            logging.info(f"Successfully retrieved credential for service '{service_name}', user '{username}'.")
            return secret
        else:
            logging.info(f"No stored credential found for service '{service_name}', user '{username}'.")
            return None
    except keyring.errors.KeyringError as e:
        logging.error(f"Failed to retrieve credential for service '{service_name}', user '{username}': {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error retrieving credential: {e}", exc_info=True)
        return None

def delete_credential(service_name: str, username: str):
    """Deletes a credential from the OS keyring."""
    try:
        keyring.delete_password(f"{AGENT_CREDENTIAL_SERVICE}_{service_name}", username)
        logging.info(f"Successfully deleted credential for service '{service_name}', user '{username}'.")
        return True
    except keyring.errors.PasswordDeleteError as e:
        # This specific error might mean it didn't exist, which is okay
        logging.warning(f"Could not delete credential for service '{service_name}', user '{username}' (may not exist): {e}")
        return False
    except keyring.errors.KeyringError as e:
        logging.error(f"Failed to delete credential for service '{service_name}', user '{username}': {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error deleting credential: {e}", exc_info=True)
        return False

