from pathlib import Path
import os
import sys

project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)
from config.config import credentials_type,log_mode,logger_name
from logger import Logger
import dotenv
if credentials_type =='env':
    dotenv.load_dotenv(os.path.join(project_root, '.env'))
if credentials_type =='docker_env':
    dotenv.load_dotenv()
# Create a singleton instance
logger = Logger(logger_name=logger_name,log_mode=log_mode,logs_dir=project_root)


def get_docker_credentials():
    """
    Retrieve credentials based on configuration.
    
    Returns:
    - tuple: (username, password)
    """
    try:
        with open('/run/secrets/username', 'r') as f:
                username = f.read().strip()
        with open('/run/secrets/password', 'r') as f:
                password = f.read().strip()
        logger.info("Docker secrets loaded")
        logger.info("Docker secrets loaded successfully")
        return username, password
    except FileNotFoundError:
        logger.error("Docker secrets file not found")
        return None, None
    
def check_pass_name(username:str,password:str):
    """
    Validate user credentials against predefined credentials.

    inputs:
    - username (str): Provided username
    - password (str): Provided password
    
    Returns:
    - bool: True if credentials match, False otherwise
    """
    logger.info(f"Attempting to validate credentials for username: {username}")
    if credentials_type =='env' or 'docker_env': 
        secret_username,secret_password = os.getenv('USERNAME'), os.getenv('PASSWORD')
    else:
        secret_username,secret_password =get_docker_credentials()

    if username==secret_username and password==secret_password:
        logger.info(f"User credentials validated successfully for username: {username}")
        return True
    else:
        logger.warning(f"Authentication failed for username: {username}")
        return False
