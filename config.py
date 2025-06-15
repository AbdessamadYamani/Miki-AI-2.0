from google.genai import types
import logging 
import os
import sys
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
YELLOW = '\033[93m'

try:
    GOOGLE_SEARCH_TOOL = Tool(
    google_search = GoogleSearch()
)

except ImportError:
    GOOGLE_SEARCH_TOOL = None
    logging.error("Failed to import google.genai.types. GoogleSearchRetrieval tool will not be available.")

# Initialize with no API key - users must provide their own key
API_KEY = None

from google import genai as genait

MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Corrected model name

AGENT_CREDENTIAL_SERVICE = "GeminiPCAgent"

CONFIRM_RISKY_COMMANDS = True
SYSTEM_PROMPT_PLANNER = "You are an expert PC Automation Assistant. Your goal is to determine the single next logical action or a tight sequence of actions (`multi_action`) to progress towards the user's objective. You output actions via function calls. If the user's request is ambiguous or requires missing information (like credentials not found), ask a clarifying question using the 'ask_user' function first. Prioritize non-visual actions (shell, python, shortcuts) over visual ones (click, type) when possible. Learn from history and critiques. Handle conversational inputs appropriately. Manage the specialized agent creation flow."

# Directory paths
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".pc_agent_cache")
DEBUG_DIR = os.path.join(CACHE_DIR, "debug")
LOG_FILE = os.path.join(DEBUG_DIR, "agent_log.txt")
CHROMA_DATA_PATH = os.path.join(CACHE_DIR, "chroma_db_store")
SHORTCUT_CACHE_FILE = os.path.join(CACHE_DIR, "shortcuts_cache.json")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHORTCUT_DEBUG_DIR = r"\debug"

# Create necessary directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(CHROMA_DATA_PATH, exist_ok=True)
os.makedirs(SHORTCUT_DEBUG_DIR, exist_ok=True)

# Initialize global variables
client = None
model = None
critic_model = None

def configure_gemini_api(api_key: str) -> bool:
    """Configure Gemini API with the given key and update global variables"""
    global client, model, critic_model
    try:
        logging.info("Starting Gemini API configuration...")
        logging.info(f"API Key length: {len(api_key)}")
        logging.info(f"API Key prefix: {api_key[:5]}...")
        
        if not api_key or len(api_key.strip()) == 0:
            logging.error("Empty API key provided")
            return False
            
        logging.info("Configuring genai...")
        genai.configure(api_key=api_key)
        
        logging.info("Creating client...")
        client = genait.Client(api_key=api_key)
        
        logging.info(f"Creating model with name: {MODEL_NAME}")
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        logging.info("Setting critic_model...")
        critic_model = model
        
        logging.info("Testing model with simple generation...")
        test_response = model.generate_content("Test")
        if test_response and test_response.candidates:
            logging.info("Model test successful")
        else:
            logging.error("Model test failed - no response generated")
            raise Exception("Model test failed - no response generated")
            
        print(f"{GREEN}[OK] Gemini API configured successfully.{RESET}")
        return True
    except Exception as e:
        logging.error(f"Error configuring Gemini API: {str(e)}", exc_info=True)
        print(f"{RED}[ERROR] Error configuring Gemini API: {e}. Please ensure API_KEY is valid.{RESET}")
        print(e)
        client = None
        model = None
        critic_model = None
        return False

def get_model():
    """Get the current model instance, ensuring it's properly initialized"""
    global model
    if model is None:
        if API_KEY:
            logging.info("Model is None, attempting to reconfigure with existing API key")
            if not configure_gemini_api(API_KEY):
                logging.error("Failed to reconfigure model with existing API key")
                return None
        else:
            logging.error("No API key available to initialize model")
            return None
    return model

def get_critic_model():
    """Get the current critic model instance, ensuring it's properly initialized"""
    global critic_model
    if critic_model is None:
        if API_KEY:
            logging.info("Critic model is None, attempting to reconfigure with existing API key")
            if not configure_gemini_api(API_KEY):
                logging.error("Failed to reconfigure critic model with existing API key")
                return None
        else:
            logging.error("No API key available to initialize critic model")
            return None
    return critic_model

def get_client():
    """Get the current client instance, ensuring it's properly initialized"""
    global client
    if client is None:
        if API_KEY:
            logging.info("Client is None, attempting to reconfigure with existing API key")
            if not configure_gemini_api(API_KEY):
                logging.error("Failed to reconfigure client with existing API key")
                return None
        else:
            logging.error("No API key available to initialize client")
            return None
    return client

if API_KEY:
    logging.info("API_KEY found in config, attempting to configure...")
    success = configure_gemini_api(API_KEY)
    if success:
        logging.info("Initial API configuration succeeded")
    else:
        logging.error("Initial API configuration failed")
else:
    logging.warning("No API_KEY found in config")
    print(f"{YELLOW}[WARNING] Gemini API key not configured. Please enter your API key in the chat interface.{RESET}")

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE, encoding='utf-8'),
                        logging.StreamHandler(sys.stdout) 
                    ])

# Initialize ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    default_ef = embedding_functions.DefaultEmbeddingFunction()

    task_executions_collection = chroma_client.get_or_create_collection(
        name="task_executions",
        embedding_function=default_ef,
        metadata={"hnsw:space": "cosine"} # Optional: specify distance metric
    )
    reinforcements_collection = chroma_client.get_or_create_collection(
        name="reinforcements",
        embedding_function=default_ef,
        metadata={"hnsw:space": "cosine"}
    )
    user_task_structures_collection = chroma_client.get_or_create_collection(
        name="user_task_structures",
        embedding_function=default_ef,
        metadata={"hnsw:space": "cosine"}
    )
    logging.info(f"ChromaDB initialized. Collections 'task_executions', 'reinforcements', and 'user_task_structures' ready.")
    logging.info(f"ChromaDB data path: {CHROMA_DATA_PATH}")
    logging.info(f"Task executions collection count: {task_executions_collection.count()}")
    logging.info(f"Reinforcements collection count: {reinforcements_collection.count()}")
    logging.info(f"User task structures collection count: {user_task_structures_collection.count()}")
    print(f"{GREEN}[OK] ChromaDB initialized successfully.{RESET}")

except Exception as e:
    logging.error(f"FATAL: Failed to initialize ChromaDB client or collections: {e}", exc_info=True)
    print(f"{RED}[ERROR] Failed to initialize ChromaDB: {e}{RESET}")
    chroma_client = None 
    task_executions_collection = None
    reinforcements_collection = None
    user_task_structures_collection = None