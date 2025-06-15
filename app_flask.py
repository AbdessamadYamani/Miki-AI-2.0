from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import logging
import json
from typing import List, Dict, Optional,  Any, Generator 
from datetime import datetime, timezone 
import subprocess
import threading
import re 
import uuid 
from config import model,LOG_FILE,DEBUG_DIR,chroma_client,default_ef,model
from tools.shortcuts_tool import load_shortcuts_cache
from task_exec.tasks_management import save_user_task_structure, load_user_task_structures, update_user_task_structure, delete_user_task_structure, retrieve_user_task_structure
from utils.reinforcement_util import analyze_feedback_and_generate_reinforcements
from task_exec.task_executor import iterative_task_executor
from agents.ai_agent import UIAgent

ui_agent = UIAgent(model)
from tools.actions import chat_with_user
try:

    load_shortcuts_cache()
except ImportError as e:
    print(f"Error importing from vision.py: {e}")
    print("Ensure vision.py is in the same directory or sys.path is configured correctly.")
    sys.exit(1)
except Exception as e:
    print(f"Error during initial setup from vision.py: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO for production, DEBUG can be too verbose
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',  # Added filename and lineno
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),  # Ensure UTF-8 for log file
        logging.StreamHandler(sys.stdout)  # Keep logging to stdout
    ]
)

# Filter out werkzeug logs
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Add a test log message to verify logging is working
logging.info("=== Flask application started with logging enabled ===")
logging.info(f"Log file location: {LOG_FILE}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.urandom(24) # Added secret key

try:
    if chroma_client and default_ef:
        agent_task_sessions_collection = chroma_client.get_or_create_collection(
            name="agent_task_sessions_history", # Distinct name for task session history
            embedding_function=default_ef,
            metadata={"hnsw:space": "cosine"}
        )
        logging.info("ChromaDB collection 'agent_task_sessions_history' initialized for AgentState.")
    else:
        raise ImportError("chroma_client or default_ef not available from vision.py")
except ImportError:
    logging.critical("Failed to initialize ChromaDB for AgentState. Task session history will not be persistent.")
    agent_task_sessions_collection = None # Fallback
class TaskSession:
    def __init__(self, task_id: str, task_name: str, start_time: datetime):
        self.task_id = task_id
        self.task_name = task_name
        self.start_time = start_time
        self.end_time: Optional[datetime] = None # Time when task was last active (paused or completed)
        self.status = "active"  # active, paused, completed
        # Store conversation history and agent thoughts within the session
        # This replaces the local history list in iterative_task_executor
        self.conversation_history: List[Dict] = []
        self.agent_thoughts: List[Dict] = []
        self.generator: Optional[Generator] = None # Each task session has its own generator
        self.execution_log: str = "" # This will store the formatted results string
        self.action_failure_counts: Dict[str, int] = {} # Track consecutive failures per action type
        self.youtube_references: List[Dict] = [] # Store YouTube references for the current task
        self.iteration_count: int = 0 # To track iterations for resuming
        self.total_tokens: Dict[str, int] = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0} # New
        self.initial_planning_done: bool = False # To track if initial planning/user interaction is done
        logging.info(f"[TaskSession.__init__] Initialized task {task_id} with zero token counts: {self.total_tokens}")

    def to_dict(self) -> Dict:
        logging.info(f"[TaskSession.to_dict] Task {self.task_id}: Converting to dict with token counts: {self.total_tokens}")
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "start_time": self.start_time.astimezone(timezone.utc).isoformat(), 
            "end_time": self.end_time.astimezone(timezone.utc).isoformat() if self.end_time else None,
            "status": self.status,
            "conversation_history": self.conversation_history,
            "agent_thoughts": self.agent_thoughts,
            "execution_log": self.execution_log,
            "youtube_references": self.youtube_references,
            "action_failure_counts": self.action_failure_counts,
            "total_tokens": self.total_tokens, 
            "iteration_count": self.iteration_count,
            "initial_planning_done": self.initial_planning_done,
        }

    def _accumulate_tokens(self, new_tokens: Dict[str, int]): # Helper method
        if not isinstance(new_tokens, dict):
            logging.error(f"[Token Accumulation] Invalid token data type: {type(new_tokens)}")
            return

        # Allow for missing keys, default to 0
        prompt_tokens_new = new_tokens.get("prompt_tokens", 0)
        candidates_tokens_new = new_tokens.get("candidates_tokens", 0)
        total_tokens_new = new_tokens.get("total_tokens", 0)

        if not (isinstance(prompt_tokens_new, int) and isinstance(candidates_tokens_new, int) and isinstance(total_tokens_new, int)):
            logging.error(f"[Token Accumulation] Invalid token type or missing required token count keys in: {new_tokens}")
            return # Do not proceed if types are wrong after defaulting

        try:
            logging.debug(f"[Token Accumulation] Current counts for task {self.task_id}: {self.total_tokens}")
            logging.debug(f"[Token Accumulation] Adding new counts: {{'prompt_tokens': {prompt_tokens_new}, 'candidates_tokens': {candidates_tokens_new}, 'total_tokens': {total_tokens_new}}}")

            # Ensure self.total_tokens has the base structure
            if not isinstance(self.total_tokens, dict): # Should not happen if initialized correctly
                self.total_tokens = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

            self.total_tokens["prompt_tokens"] = self.total_tokens.get("prompt_tokens", 0) + prompt_tokens_new
            self.total_tokens["candidates_tokens"] = self.total_tokens.get("candidates_tokens", 0) + candidates_tokens_new
            self.total_tokens["total_tokens"] = self.total_tokens.get("total_tokens", 0) + total_tokens_new

            logging.info(f"[Token Accumulation] New total for task {self.task_id}: {self.total_tokens}")
        except Exception as e:
            logging.error(f"[Token Accumulation] Error accumulating tokens for task {self.task_id}: {e}")

class AgentState:
    def __init__(self):
        self.current_task: Optional[TaskSession] = None
        
        self.is_task_running = False 
        self.task_is_paused = False 
        self.current_thought: Optional[str] = None 
        self.thought_history: List[Dict] = [] 
        
        self.pending_new_task_decision: Optional[Dict[str, Any]] = None
        self.status_message: str = 'Ready for new task'  # Add status message attribute

    def _save_session_to_chromadb(self, session: TaskSession):
        if not agent_task_sessions_collection:
            print("ERROR: ChromaDB collection not available. Cannot save session.")
            return
        try:
            print(f"\n[ChromaDB] Saving session {session.task_id}")
            print(f"[ChromaDB] Token counts: {session.total_tokens}")
            
            # Validate token counts before saving
            if not isinstance(session.total_tokens, dict) or not all(key in session.total_tokens for key in ["prompt_tokens", "candidates_tokens", "total_tokens"]):
                logging.error(f"[ChromaDB] Invalid token count format for task {session.task_id}: {session.total_tokens}")
                session.total_tokens = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
            
            metadata = {
                "task_id": session.task_id,
                "task_name": session.task_name,
                "start_time": session.start_time.astimezone(timezone.utc).isoformat(),
                "end_time": session.end_time.astimezone(timezone.utc).isoformat() if session.end_time else "",
                "status": session.status,
                "execution_log": session.execution_log
            } 
            metadata["total_tokens"] = json.dumps(session.total_tokens)
            metadata["iteration_count"] = session.iteration_count
            metadata["initial_planning_done"] = session.initial_planning_done
            
            print(f"[ChromaDB] Token counts in metadata: {metadata['total_tokens']}")
            
            metadata["conversation_history"] = json.dumps(session.conversation_history)
            metadata["agent_thoughts"] = json.dumps(session.agent_thoughts)
            metadata["youtube_references"] = json.dumps(session.youtube_references)

            agent_task_sessions_collection.upsert(
                ids=[session.task_id],
                documents=[session.task_name],  
                metadatas=[metadata]
            )
            print("[ChromaDB] Successfully saved session\n")
        except Exception as e:
            print(f"ERROR saving to ChromaDB: {e}")

    def _load_session_from_chromadb(self, task_id: str) -> Optional[TaskSession]:
        if not agent_task_sessions_collection:
            logging.error("ChromaDB collection for task sessions not available. Cannot load session.")
            return None
        try:
            results = agent_task_sessions_collection.get(ids=[task_id], include=['metadatas'])
            if results and results['ids'] and results['metadatas']:
                session_data = results['metadatas'][0]

                start_time = datetime.fromisoformat(session_data["start_time"].replace("Z", "+00:00"))
                end_time = datetime.fromisoformat(session_data["end_time"].replace("Z", "+00:00")) if session_data["end_time"] else None

                session = TaskSession(
                    task_id=session_data["task_id"],
                    task_name=session_data["task_name"],
                    start_time=start_time
                )
                session.end_time = end_time
                session.status = session_data["status"]
                session.execution_log = session_data["execution_log"]

                try:
                    if "total_tokens" in session_data and session_data["total_tokens"]: # Check if not empty
                        loaded_tokens = json.loads(session_data["total_tokens"])
                        if isinstance(loaded_tokens, dict) and all(key in loaded_tokens for key in ["prompt_tokens", "candidates_tokens", "total_tokens"]):
                            session.total_tokens = loaded_tokens
                            logging.info(f"[ChromaDB] Loaded token counts for task {task_id}: {session.total_tokens}")
                        else:
                            logging.warning(f"[ChromaDB] Invalid token count format in metadata for task {task_id}: {loaded_tokens}, using defaults.")
                            session.total_tokens = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
                    else:
                        logging.warning(f"[ChromaDB] No token counts found or empty in metadata for task {task_id}, using defaults")
                        session.total_tokens = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
                except json.JSONDecodeError as e:
                    logging.error(f"[ChromaDB] Error decoding token counts JSON for task {task_id}: {e}. Metadata value: '{session_data.get('total_tokens', 'MISSING')}'")
                    session.total_tokens = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
                except Exception as e: # Catch other potential errors during token loading
                    logging.error(f"[ChromaDB] Generic error loading token counts for task {task_id}: {e}")
                    session.total_tokens = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

                session.conversation_history = json.loads(session_data["conversation_history"])
                session.agent_thoughts = json.loads(session_data["agent_thoughts"])
                session.youtube_references = json.loads(session_data.get("youtube_references", "[]"))
                session.iteration_count = session_data.get("iteration_count", 0)
                session.initial_planning_done = session_data.get("initial_planning_done", False)

                logging.debug(f"Loaded task session {task_id} from ChromaDB.")
                return session
            else:
                logging.warning(f"Task session {task_id} not found in ChromaDB.")
                return None
        except Exception as e:
            logging.error(f"Error loading task session {task_id} from ChromaDB: {e}", exc_info=True)
            return None

    def start_new_task(self, task_name: str) -> TaskSession:
        if self.current_task and self.current_task.status == "active":
            self.pause_current_task() 

        task_id = str(uuid.uuid4()) 
        session = TaskSession(task_id, task_name, datetime.now(timezone.utc)) 
        
        self.current_task = session
        self.is_task_running = True 
        self.task_is_paused = False 
        self.status_message = f'Starting task: {task_name}'
        
        self._save_session_to_chromadb(session)
        logging.info(f"Started new task: {task_id} - '{task_name}'")
        return session
            
    def pause_current_task(self) -> Optional[TaskSession]:
        if self.current_task:
            logging.info(f"Pausing task: {self.current_task.task_id} - '{self.current_task.task_name}'")
            if self.current_task.status != "paused": 
                self.current_task.status = "paused"
                self.current_task.end_time = datetime.now(timezone.utc)
                self.is_task_running = False
                self.task_is_paused = True
                self.status_message = f'Task paused: {self.current_task.task_name}'
                self._save_session_to_chromadb(self.current_task)
            return self.current_task
        return None

    def resume_task(self, task_id: str) -> Optional[TaskSession]:
        session_to_resume = self._load_session_from_chromadb(task_id)
        if session_to_resume:
            if self.current_task and self.current_task.task_id != task_id and self.current_task.status == "active":
                self.pause_current_task()

            logging.info(f"Resuming task: {session_to_resume.task_id} - '{session_to_resume.task_name}'")
            
            self.current_task = session_to_resume
            
            if session_to_resume.status == "paused":
                self.is_task_running = True
                self.task_is_paused = True
                session_to_resume.status = "paused"  
                self.status_message = f'Task resumed (paused): {session_to_resume.task_name}'
            else: # If it was completed or failed, resuming effectively makes it active again for new input
                session_to_resume.status = "active"
                self.is_task_running = True
                self.task_is_paused = False
                self.status_message = f'Task resumed (active): {session_to_resume.task_name}'
            
            session_to_resume.end_time = None # Clear end time on resume
            self._save_session_to_chromadb(session_to_resume)
            
            return session_to_resume
        logging.warning(f"Attempted to resume non-existent task_id: {task_id}")
        return None

    def add_thought(self, thought: str, type: str = "thinking"): 
        if self.current_task:
            thought_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content": thought,
                "type": type
            }
            
            self.current_task.agent_thoughts.append(thought_entry)
            
            # If this is a reasoning thought, also add it to the conversation history
            if type == "reasoning": # Only add to conversation_history if explicitly type "reasoning"
                # Extract the reasoning part if it's prefixed with "Reasoning:"
                reasoning_content = thought.split("Reasoning:", 1)[1].strip() if thought.startswith("Reasoning:") else thought

                # Format the reasoning as a JSON plan
                json_plan = {
                    "reasoning": reasoning_content
                }
                formatted_content = f"--- Agent Reasoning Snippet at {datetime.now(timezone.utc).isoformat()} ---\n{json.dumps(json_plan, indent=2)}" # Changed title

                # Add to conversation history
                self.current_task.conversation_history.append({
                    "role": "assistant",
                    "content": formatted_content
                })
            
            self.current_thought = thought 
            self._save_session_to_chromadb(self.current_task) # Save after adding thought
            logging.debug(f"Thought ({type}): {thought}")

    def get_task_history_list(self) -> List[Dict]: 
        if not agent_task_sessions_collection:
            logging.error("ChromaDB collection for task sessions not available. Cannot get history.")
            return []
        try:
            results = agent_task_sessions_collection.get(include=['metadatas'])
            history_list = []
            if results and results['metadatas']:
                for session_data in results['metadatas']:
                    # Ensure all expected keys are present or provide defaults
                    history_item = {
                        'task_id': session_data.get('task_id', 'N/A'),
                        'task_name': session_data.get('task_name', 'Unnamed Task'),
                        'start_time': session_data.get('start_time', ''),
                        'end_time': session_data.get('end_time', None),
                        'status': session_data.get('status', 'unknown')
                    }
                    history_list.append(history_item)
            return sorted(history_list, key=lambda x: x.get('start_time', ''), reverse=True)
        except Exception as e:
            logging.error(f"Error fetching task history from ChromaDB: {e}", exc_info=True)
            return []

    def find_similar_task_sessions(self, query_text: str, n_results: int = 3, similarity_threshold: float = 0.70) -> List[Dict]: 
        if not agent_task_sessions_collection:
            logging.error("ChromaDB collection for task sessions not available. Cannot find similar sessions.")
            return []
        if not query_text:
            return []
        try:
            # Ensure n_results is not greater than the number of items in the collection
            collection_count = agent_task_sessions_collection.count()
            if collection_count == 0:
                return []
            
            query_results = agent_task_sessions_collection.query(
                query_texts=[query_text],
                n_results=min(n_results * 2, collection_count), # Fetch a bit more to filter by status
                include=['metadatas', 'distances']
            )

            similar_sessions = []
            if query_results and query_results['ids'] and query_results['ids'][0]:
                for i in range(len(query_results['ids'][0])):
                    task_id = query_results['ids'][0][i]
                    metadata = query_results['metadatas'][0][i]
                    distance = query_results['distances'][0][i]

                    if distance < similarity_threshold: 
                        # Filter out tasks that are already completed or failed, as they might not be good candidates for "resuming" in the context of a new, similar query.
                        # However, for "new task" prompt, we might want to show them as examples of past work.
                        # For now, let's keep the original logic of showing resumable (active/paused) tasks.
                        if metadata.get('status') not in ['completed', 'failed']:
                            similar_sessions.append({
                                "task_id": task_id,
                                "task_name": metadata.get("task_name", "Unnamed Task"),
                                "status": metadata.get("status", "unknown"),
                                "start_time": metadata.get("start_time", ""),
                                "distance": distance
                            })
            
            # Sort by distance (closer is better), then by start_time (more recent is better)
            similar_sessions.sort(key=lambda x: (x['distance'], x.get('start_time', '')), reverse=False) # distance ascending, start_time descending if needed
            logging.info(f"Found {len(similar_sessions)} similar, resumable task sessions for query '{query_text[:50]}...' (threshold: {similarity_threshold}).")
            return similar_sessions[:n_results]
        except Exception as e:
            logging.error(f"Error finding similar task sessions in ChromaDB: {e}", exc_info=True)
            return []

# This should be done once when the app starts
agent_state = AgentState()
def format_results_for_display(results: Any) -> str:
    if not results:
        return "No actions were executed."
    
    lines = ["=== Execution Log ==="]

    
    if isinstance(results, dict):
        if "results" in results:
            
            lines.append("Note: Input was a dict, formatting 'results' key.")
            results = results["results"]
        else:
            return str(results) 
    
    
    if not isinstance(results, list): 
        return f"Unexpected result format: {type(results)}. Content: {str(results)}"
    
    for i, result_item in enumerate(results, 1):
        try:
            if isinstance(result_item, tuple) and len(result_item) >= 3:
                action, success, message = result_item[:3] # Unpack only the first 3 for display
                action_type = action.get('action_type', 'Unknown Action') if isinstance(action, dict) else str(action)
                status_icon = "✅" if success else "❌"
                
                message_str = str(message)
                message_short = (message_str[:250] + '...') if len(message_str) > 253 else message_str
                lines.append(f"{i}. [{action_type}] {status_icon}\n   {message_short}")
            elif isinstance(result_item, dict): 
                action_type = result_item.get('action_type', result_item.get('type', 'Unknown Event'))
                status_icon = "✅" if result_item.get('success', True) else "❌" 
                message = result_item.get('message', result_item.get('reason', str(result_item)))
                message_short = (message[:250] + '...') if len(message) > 253 else message
                lines.append(f"{i}. [{action_type}] {status_icon}\n   {message_short}")
            else:
                lines.append(f"{i}. {str(result_item)}")
        except Exception as e: 
            logging.error(f"Error formatting result item {i}: {e}. Item: {result_item}", exc_info=True)
            lines.append(f"{i}. [Error formatting result item] {str(result_item)}")
    
    return "\n".join(lines)
def get_ui_update_state():
    """Get the current UI state."""
    global agent_state
    
    # Get current token counts
    current_tokens = agent_state.current_task.total_tokens if agent_state.current_task else {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    
    # Get the latest reasoning if available
    latest_reasoning = None
    if agent_state.current_task and agent_state.current_task.agent_thoughts:
        # Look for the most recent reasoning thought
        for thought_entry in reversed(agent_state.current_task.agent_thoughts):
            thought_type = thought_entry.get('type')
            thought_content = thought_entry.get('content', '')

            if thought_type == "planning_json":
                latest_reasoning = thought_content
                break
            elif thought_type in ['planning', 'reasoning', 'reasoning_selection', 'critique_reasoning', 'assessment_reasoning', 'replan_reasoning']:
                if "Reasoning: " in thought_content:
                    latest_reasoning = thought_content.split("Reasoning: ", 1)[-1].strip()
                elif thought_type == 'planning' and ", Reasoning: " in thought_content:
                    latest_reasoning = thought_content.split(", Reasoning: ", 1)[-1].strip()
                elif thought_content:
                    latest_reasoning = thought_content
                break
        if not latest_reasoning and agent_state.current_task.agent_thoughts:
            latest_reasoning = agent_state.current_task.agent_thoughts[-1].get('content', '')

    # Get the last thought timestamp from the request
    last_thought = request.args.get('last_thought', '')
    
    # Filter thoughts to only include new ones since last update
    agent_thoughts = []
    if agent_state.current_task and agent_state.current_task.agent_thoughts:
        if last_thought:
            agent_thoughts = [
                thought for thought in agent_state.current_task.agent_thoughts
                if thought['timestamp'] > last_thought
            ]
        else:
            agent_thoughts = agent_state.current_task.agent_thoughts

    ui_state = {
        "userInputInteractive": not agent_state.is_task_running,
        "sendInteractive": not agent_state.is_task_running,
        "stopInteractive": agent_state.is_task_running,
        "stopVisible": agent_state.is_task_running,
        "continueInteractive": agent_state.task_is_paused,
        "continueVisible": agent_state.task_is_paused,
        "statusMessage": agent_state.status_message,
        "conversationHistory": agent_state.current_task.conversation_history if agent_state.current_task else [],
        "executionLog": agent_state.current_task.execution_log if agent_state.current_task else "",
        "currentTaskName": agent_state.current_task.task_name if agent_state.current_task else "None",
        "currentTaskId": agent_state.current_task.task_id if agent_state.current_task else None,
        "taskHistory": agent_state.get_task_history_list(),
        "totalTokens": current_tokens,
        "latestReasoning": latest_reasoning,
        "isThinking": agent_state.is_task_running,
        "showThoughts": True,
        "agentThoughts": agent_thoughts
    }
    
    return ui_state

@app.route('/')
def index():
    # Pass initial state to the template
    # Determine initial task to display if any.
    if agent_state.current_task is None and agent_task_sessions_collection:
        history_summary = agent_state.get_task_history_list() # Gets sorted list
        if history_summary:
            # Try to find the first non-completed/failed task from summary
            active_or_paused_task_info = next((task_info for task_info in history_summary if task_info.get('status') not in ["completed", "failed"]), None)
            
            task_to_load_id = None
            if active_or_paused_task_info:
                task_to_load_id = active_or_paused_task_info.get('task_id')
            elif history_summary: # Fallback to the absolute most recent if all are done
                task_to_load_id = history_summary[0].get('task_id')

            if task_to_load_id:
                agent_state.current_task = agent_state._load_session_from_chromadb(task_to_load_id)
                if agent_state.current_task:
                    # If loaded, ensure its status is appropriate (e.g., if "active", maybe set to "paused" for UI)
                    if agent_state.current_task.status == "active": # If it was active, show as paused
                        agent_state.current_task.status = "paused"
                        agent_state.is_task_running = True # It was running
                        agent_state.task_is_paused = True  # Now it's paused for UI

    initial_ui_state = get_ui_update_state()
    return render_template('index.html',
                           log_file=LOG_FILE, debug_dir=DEBUG_DIR,
                           initial_ui_state_json=json.dumps(initial_ui_state) # Pass full state
                           )

@app.route('/send_message', methods=['POST'])
def send_message():
    global agent_state
    try:
        data = request.json
        user_input_raw = data.get('message') if data else None
        file_data = data.get('file') if data else None
        user_input_content = ""

        if user_input_raw is not None:
            user_input_content = user_input_raw.strip()

        # Add file information to the message if a file was uploaded
        if file_data and file_data.get('success'):
            filepath = file_data.get('filepath')
            if filepath:
                user_input_content = f"file:{filepath} {user_input_content}"

        # 1. Handle pending_new_task_decision if user provided input
        if agent_state.pending_new_task_decision and user_input_content:
            original_instruction_for_new = agent_state.pending_new_task_decision["original_instruction"]
            similar_options = agent_state.pending_new_task_decision["similar_sessions"]
            user_choice_raw = user_input_content.lower().strip()

            agent_state.pending_new_task_decision = None # Clear the pending decision

            resumed_task_id = None
            if user_choice_raw.startswith("resume "):
                try:
                    potential_task_id = user_choice_raw.split(" ", 1)[1].strip()
                    if any(opt['task_id'] == potential_task_id for opt in similar_options):
                        resumed_task_id = potential_task_id
                except IndexError: pass

            if resumed_task_id:
                logging.info(f"User chose to resume task: {resumed_task_id} for original query '{original_instruction_for_new}'")
                resumed_session = agent_state.resume_task(resumed_task_id)
                if resumed_session:
                    # Add the original user input that triggered the choice, and a system message
                    resumed_session.conversation_history.append({"role": "user", "content": user_input_raw}) # The 'resume TASK_ID' or 'new'
                    resumed_session.conversation_history.append({"role": "system", "content": f"Task resumed. The original new query was: '{original_instruction_for_new}'. Please provide input for this resumed task, or type 'continue' if the original query should be used as the next input."})
                    user_input_content = "" # Clear user_input_content as it was a command, not direct input for the task yet
                    agent_state.task_is_paused = True # Paused, waiting for user to give actual input for the resumed task or 'continue'
                    agent_state.is_task_running = True # It's active in the sense that it's the current task
                else:
                    logging.error(f"Failed to resume task {resumed_task_id}. Starting new for '{original_instruction_for_new}'.")
                    agent_state.start_new_task(original_instruction_for_new)
                    agent_state.current_task.conversation_history.append({"role": "user", "content": user_input_raw}) # The 'resume TASK_ID' or 'new'
                    agent_state.current_task.conversation_history.append({"role": "system", "content": "Failed to resume, created new task."})
                    user_input_content = original_instruction_for_new # This becomes the first input for the new task
            elif "new" in user_choice_raw:
                logging.info(f"User chose to create a new task for: '{original_instruction_for_new}'")
                agent_state.start_new_task(original_instruction_for_new)
                agent_state.current_task.conversation_history.append({"role": "user", "content": user_input_raw}) # The 'new' command
                agent_state.current_task.conversation_history.append({"role": "system", "content": "User opted to create a new task."})
                user_input_content = original_instruction_for_new # This becomes the first input for the new task
            else: # Invalid choice
                logging.warning(f"Invalid choice '{user_choice_raw}' for similar task prompt. Defaulting to new task for '{original_instruction_for_new}'.")
                agent_state.start_new_task(original_instruction_for_new)
                agent_state.current_task.conversation_history.append({"role": "user", "content": user_input_raw}) # The invalid command
                agent_state.current_task.conversation_history.append({"role": "system", "content": "Invalid choice for similar task, created new task."})
                user_input_content = original_instruction_for_new # This becomes the first input for the new task
            
            # If after handling the choice, user_input_content is empty (e.g. task resumed and waiting for next command)
            # and the task is paused, return the UI state.
            if not user_input_content and agent_state.current_task and agent_state.current_task.status == "paused":
                 return jsonify(get_ui_update_state())

        # 2. Process new user input if no pending decision was just handled, or if it fell through
        elif user_input_content:
            if agent_state.current_task is None or agent_state.current_task.status == "completed":
                # Context: No active task, or the current one is done.
                # This is where we consider starting a genuinely new task or resuming a *different* non-active task.
                similar_sessions = agent_state.find_similar_task_sessions(user_input_content, n_results=3)
                current_task_id_if_exists = agent_state.current_task.task_id if agent_state.current_task else None
                # Filter out the current task if it's 'completed' to avoid offering to resume itself as a "similar" task.
                resumable_options = [
                    s for s in similar_sessions
                    if s['status'] not in ['completed', 'failed'] and s.get('task_id') != current_task_id_if_exists
                ]

                if resumable_options:
                    question_parts = ["I found some similar past tasks that are not finished:"]
                    for i, sess_opt in enumerate(resumable_options):
                        question_parts.append(f"{i+1}. '{sess_opt['task_name']}' (ID: {sess_opt['task_id']}, Status: {sess_opt['status']})")
                    question_parts.append(f"\nType 'resume TASK_ID' to resume one, or 'new' to create a new task for your instruction: '{user_input_content}'.")
                    question_to_ask_user = "\n".join(question_parts)

                    agent_state.pending_new_task_decision = {
                        "original_instruction": user_input_content,
                        "similar_sessions": resumable_options,
                        "question_asked_to_user": question_to_ask_user
                    }
                    logging.info(f"Offering user choice for instruction '{user_input_content}'. Similar tasks found.")
                    
                    # Add the question to the history of the current (completed) task if it exists
                    if agent_state.current_task and (not agent_state.current_task.conversation_history or \
                        agent_state.current_task.conversation_history[-1].get("content") != question_to_ask_user):
                        agent_state.current_task.conversation_history.append({
                            "role": "system", # Or "assistant"
                            "content": question_to_ask_user
                        })
                        agent_state._save_session_to_chromadb(agent_state.current_task)

                    return jsonify(get_ui_update_state())
                else:
                    # No similar resumable tasks, or user implicitly wants new. Start a new task.
                    logging.info(f"No similar resumable tasks found for '{user_input_content}'. Starting new task.")
                    agent_state.start_new_task(user_input_content)
                    # The user_input_content is the task name and also the first user message.
                    agent_state.current_task.conversation_history.append({
                        "role": "user",
                        "content": user_input_content
                    })
                    agent_state._save_session_to_chromadb(agent_state.current_task)

            elif agent_state.current_task.status == "failed":
                # Context: User is providing input to a previously FAILED task.
                logging.info(f"User provided input for a FAILED task '{agent_state.current_task.task_id}'. Re-activating.")
                agent_state.current_task.conversation_history.append({
                    "role": "user",
                    "content": user_input_content
                })
                agent_state.current_task.status = "active"  # Re-open the task
                agent_state.current_task.end_time = None     # Clear end time
                agent_state.is_task_running = False # It's not running yet, but ready for generator
                agent_state.task_is_paused = False
                agent_state.status_message = f'Re-opened failed task: {agent_state.current_task.task_name}'
                # The generator will be re-initialized later in this function.

            elif agent_state.current_task.status in ["active", "paused"]:
                # Context: User is providing input to an already active or paused task.
                # Append if it's a new message.
                if not agent_state.current_task.conversation_history or \
                   agent_state.current_task.conversation_history[-1].get("content") != user_input_content or \
                   agent_state.current_task.conversation_history[-1].get("role") != "user":
                    agent_state.current_task.conversation_history.append({
                        "role": "user",
                        "content": user_input_content
                    })
                agent_state.current_task.end_time = datetime.now(timezone.utc) # Mark activity
            # agent_state._save_session_to_chromadb(agent_state.current_task) # Save history additions before generator

        # 3. Ensure there's a current task to work with before proceeding to generator
        if agent_state.current_task is None:
            if not user_input_content and not agent_state.pending_new_task_decision:
                 logging.info("No current task, no input, and not waiting for a decision.")
                 return jsonify(get_ui_update_state())
            elif agent_state.pending_new_task_decision:
                # If we are here, it means user_input_content was empty, but a decision is pending.
                # The UI should reflect the question.
                # The logic in step 1 (pending_new_task_decision) handles non-empty input.
                # If current_task exists (e.g. was 'completed' and then we asked about similar tasks),
                # its history should already contain the question.
                logging.info("Waiting for user decision on pending new task prompt.")
                return jsonify(get_ui_update_state())

            # If user_input_content was provided, a task should have been created or re-activated.
            logging.error("CRITICAL: No current task to process input. This might happen if input was empty and no decision was pending.")
            return jsonify({"status": "error", "message": "Error: No task context after input processing."}), 500

        # 4. Generator handling logic
        next_yield = None
        # Determine input for generator: if user_input_content was just processed and led to this point, use it.
        # If it was a 'continue' or internal progression, input_for_generator might be None.
        input_for_generator = user_input_content if user_input_content else None
        # This is correct: if user_input_content drove the logic to this point, it's the input.
        # If send_message is called without user_input_content (e.g. after a 'continue' that didn't immediately yield), it's None.

        try:
            # When to initialize/re-initialize the generator:
            # - If there's no generator for the current task.
            # - If the task was 'active' but not 'running' (e.g., just re-activated from 'failed' or 'paused' with new input).
            # - If the task was 'paused' and now receives new input (user_input_content is not empty).
            # - If it's a brand new task.
            should_initialize_generator = False
            if agent_state.current_task.generator is None:
                should_initialize_generator = True
            elif agent_state.current_task.status == "active" and not agent_state.is_task_running: # Covers re-opened 'failed'
                should_initialize_generator = True
            elif agent_state.current_task.status == "paused" and input_for_generator: # Paused task gets new text input
                should_initialize_generator = True
            
            if agent_state.current_task and agent_state.current_task.status == "paused" and not input_for_generator:
                # This is a "continue" scenario where the /continue_task endpoint might not have been hit,
                # or it yielded something that brought us back here without new user text.
                # The generator should be sent None.
                logging.info(f"Task '{agent_state.current_task.task_name}' is paused, and no new input. Sending None to generator if it exists.")
                # should_initialize_generator remains as determined above. If generator exists, it will be sent None.

            if should_initialize_generator:
                logging.info(f"Initializing generator for task: {agent_state.current_task.task_name} (Status: {agent_state.current_task.status}, Running: {agent_state.is_task_running})")
                string_initial_history = [f"{msg['role']}: {msg['content']}" for msg in agent_state.current_task.conversation_history]
                agent_state.current_task.generator = iterative_task_executor(
                    original_instruction=agent_state.current_task.task_name,
                    agent_state=agent_state,
                    agent=ui_agent,
                    llm_model=model,
                    initial_history=string_initial_history
                )
                agent_state.is_task_running = True
                agent_state.task_is_paused = False
                agent_state.current_task.status = "active" # Ensure status is active

                try:
                    next_yield = next(agent_state.current_task.generator)
                    agent_state._save_session_to_chromadb(agent_state.current_task)
                except StopIteration as e:
                    handle_task_completion(agent_state.current_task, e.value)
                    return jsonify(get_ui_update_state())

            else: # Generator exists, task is running or was paused and now continuing
                log_msg_input = input_for_generator if input_for_generator is not None else "<None_explicit>"
                logging.info(f"Sending to existing generator for task '{agent_state.current_task.task_name}' (Status: {agent_state.current_task.status}, Running: {agent_state.is_task_running}): Input='{log_msg_input}'")

                agent_state.is_task_running = True
                agent_state.task_is_paused = False
                if agent_state.current_task.status == "paused": # If it was paused, mark active
                    agent_state.current_task.status = "active"
                    agent_state.current_task.end_time = None

                try:
                    next_yield = agent_state.current_task.generator.send(input_for_generator)
                    agent_state._save_session_to_chromadb(agent_state.current_task)
                except StopIteration as e:
                    handle_task_completion(agent_state.current_task, e.value) # Make sure current_task is not None
                    return jsonify(get_ui_update_state())
                except Exception as gen_e:
                    logging.error(f"Error sending input to generator: {gen_e}", exc_info=True)
                    handle_task_error(agent_state.current_task, gen_e)
                    return jsonify(get_ui_update_state())
                
            if isinstance(next_yield, dict) and "type" in next_yield:
                yield_type = next_yield["type"]
                agent_state.add_thought(f"Generator yielded: {yield_type}", type="yield")

                if yield_type == "ask_user":
                    question = next_yield.get("question", "...")
                    logging.info(f"Generator yielded 'ask_user': {question}")
                    agent_state.current_task.conversation_history.append({
                        "role": "assistant",
                        "content": f"❓ {question}"
                    })
                    agent_state.status_message = f'Waiting for user input: {agent_state.current_task.task_name}'
                    agent_state.task_is_paused = True # Task is paused waiting for user
                    agent_state.is_task_running = False # Agent is not actively thinking
                    agent_state.current_task.status = "paused"
                    agent_state._save_session_to_chromadb(agent_state.current_task)
                    return jsonify(get_ui_update_state())
                elif yield_type == "inform_user":
                    message = next_yield.get("message", "...")
                    logging.info(f"Generator yielded 'inform_user': {message[:100]}...")
                    youtube_refs = next_yield.get("youtube_references", [])
                    image_data = next_yield.get("image_data")  # Get image data if present
                    
                    message_content = {
                        "role": "assistant",
                        "content": f"ℹ️ {message}"
                    }
                    
                    if youtube_refs:
                        agent_state.current_task.youtube_references.extend(youtube_refs)
                        message_content["references"] = youtube_refs
                    
                    if image_data:
                        message_content["image_data"] = image_data
                    
                    agent_state.current_task.conversation_history.append(message_content)
                    agent_state.status_message = f'Agent informed user: {agent_state.current_task.task_name}'
                    agent_state.task_is_paused = True # Task is paused waiting for user
                    agent_state.is_task_running = False # Agent is not actively thinking
                    agent_state.current_task.status = "paused"
                    agent_state._save_session_to_chromadb(agent_state.current_task)
                    return jsonify(get_ui_update_state())
                elif yield_type == "paused":
                    logging.info("Generator yielded 'paused'. Task is now paused by agent's internal logic.")
                    agent_state.status_message = f'Task paused by agent: {agent_state.current_task.task_name}'
                    agent_state.is_task_running = False
                    agent_state.task_is_paused = True
                    agent_state.current_task.status = "paused"
                    agent_state._save_session_to_chromadb(agent_state.current_task)
                    return jsonify(get_ui_update_state())
                else:
                    logging.warning(f"Generator yielded unknown type: {yield_type}. Content: {next_yield}")
                    handle_task_error(agent_state.current_task, Exception(f"Agent yielded unknown type: {yield_type}"))
            else:
                logging.warning(f"Generator yielded unexpected value: {next_yield}. Treating as error.")
                handle_task_error(agent_state.current_task, Exception(f"Agent yielded unexpected value: {next_yield}"))

        except StopIteration as e:
            logging.info(f"Task '{agent_state.current_task.task_name}' generator finished.")
            handle_task_completion(agent_state.current_task, e.value)
        except Exception as e:
            logging.critical(f"Critical error in send_message processing block: {e}", exc_info=True)
            if agent_state.current_task:
                handle_task_error(agent_state.current_task, e)
            else:
                return jsonify({"status": "error", "message": f"Critical error: {str(e)}"}), 500
        
        return jsonify(get_ui_update_state())

    except Exception as e:
        logging.critical(f"Critical error in send_message route: {e}", exc_info=True)
        error_response = get_ui_update_state() # Try to get UI state even on error
        error_response["status"] = "error"
        # Ensure message is a string, even if e is not directly convertible
        error_response["message"] = f"Outer error: {str(e)}" if str(e) else "An unspecified outer error occurred."
        # Add more details to the log for such critical errors
        logging.error(f"Outer error in send_message. Current task ID: {agent_state.current_task.task_id if agent_state.current_task else 'None'}. Pending decision: {agent_state.pending_new_task_decision is not None}")
        return jsonify(error_response), 500


@app.route('/new_task', methods=['POST'])
def new_task_route(): 
    global agent_state
    data = request.get_json()
    task_name = data.get('task_name', f'Task_{uuid.uuid4().hex[:6]}') 
    
    
    session = agent_state.start_new_task(task_name)
    
    # Explicitly set the task to paused and ensure is_task_running reflects this
    session.status = "paused"
    agent_state.task_is_paused = True
    agent_state.is_task_running = False # Ensure it's not considered "running" yet
    
    session.conversation_history.append({
        "role": "system",
        "content": f"New task '{task_name}' created. Please type your instruction and click 'Send' to begin."
    })
    agent_state._save_session_to_chromadb(session) 
    logging.info(f"API: New task '{task_name}' (ID: {session.task_id}) created and set as current, paused.")
    return jsonify({
        "status": "success",
        "message": f"New task '{task_name}' started.",
        **get_ui_update_state() 
    })

@app.route('/stop_task', methods=['POST'])
def stop_task_route(): # Renamed
    global agent_state
    if agent_state.current_task:
        logging.info(f"API: Stop command received for task: {agent_state.current_task.task_id}")
        session = agent_state.pause_current_task() # This also saves to ChromaDB
        if session:
             session.conversation_history.append({"role": "system", "content": "Task manually paused by user."})
             agent_state._save_session_to_chromadb(session) # Save the history addition
        return jsonify({
            "status": "success",
            "message": f"Task '{session.task_name if session else 'N/A'}' pause signal sent.",
            **get_ui_update_state() 
        })
    return jsonify({"status": "error", "message": "No active task to stop.", **get_ui_update_state()}), 400

@app.route('/continue_task', methods=['POST'])
def continue_task_route(): 
    global agent_state
    if agent_state.current_task and agent_state.current_task.status == "paused":
        logging.info(f"API: Continue command received for task: {agent_state.current_task.task_id}")
        agent_state.current_task.status = "active"
        agent_state.task_is_paused = False 
        agent_state.is_task_running = True 
        agent_state.current_task.end_time = None 
        agent_state.current_task.conversation_history.append({"role": "system", "content": "Task continued by user."})
        agent_state._save_session_to_chromadb(agent_state.current_task) 


        
        if agent_state.current_task and agent_state.current_task.generator:
            logging.info(f"Sending None to resume generator for task: {agent_state.current_task.task_name}")
            try:
                next_yield = agent_state.current_task.generator.send(None) 

                
                if isinstance(next_yield, dict) and "type" in next_yield:
                    yield_type = next_yield["type"]
                    agent_state.add_thought(f"Generator yielded after continue: {yield_type}", type="yield")
                    if yield_type == "ask_user":
                        question = next_yield.get("question", "...")
                        agent_state.current_task.conversation_history.append({"role": "assistant", "content": question})
                        agent_state.task_is_paused = True 
                    elif yield_type == "inform_user":
                         message = next_yield.get("message", "...")
                         agent_state.current_task.conversation_history.append({"role": "assistant", "content": message})
                         agent_state.task_is_paused = True 
                    elif yield_type == "paused":
                         logging.info("Generator yielded 'paused' again immediately after continue.")
                         agent_state.current_task.status = "paused" 
                         agent_state.current_task.end_time = datetime.now(timezone.utc) 
                         agent_state.is_task_running = False
                         agent_state.task_is_paused = True
                    
                

            except StopIteration as e:
                logging.info("Generator finished after continue/resume.")
                handle_task_completion(agent_state.current_task, e.value) 
            except Exception as gen_e:
                logging.error(f"Error resuming generator on continue: {gen_e}", exc_info=True)
                handle_task_error(agent_state.current_task, gen_e) 
        else:
            logging.info(f"Task {agent_state.current_task.task_id} unpaused. Waiting for next user input to start/resume generator.")
        
        
        if not (agent_state.current_task.status == "completed" or agent_state.current_task.status == "failed"):
            agent_state._save_session_to_chromadb(agent_state.current_task)
        return jsonify(get_ui_update_state())
    
    logging.warning("Continue task called but no current task is paused.")
    return jsonify({"status":"error", "message": "No task to continue or task not paused.", **get_ui_update_state()}), 400


# --- User Task Structure Management API Endpoints (Unchanged from previous) ---
@app.route('/tasks/list', methods=['GET'])
def list_tasks():
    try:
        tasks = load_user_task_structures() # From vision.py
        # Transform the data to match frontend expectations
        formatted_tasks = []
        for task in tasks:
            formatted_task = {
                "id": task["id"],
                "name": task["task_name"],
                "preview": task["plan_text"][:100] + "..." if len(task["plan_text"]) > 100 else task["plan_text"],
                "full_plan": task["plan_text"]
            }
            formatted_tasks.append(formatted_task)
        return jsonify({"success": True, "tasks": formatted_tasks, "message": f"Loaded {len(formatted_tasks)} saved task structures."})
    except Exception as e:
        logging.error(f"Error in /tasks/list: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/tasks/save', methods=['POST'])
def save_task_api():
    try:
        data = request.json
        task_name = data.get('name')
        plan_text = data.get('plan') # Assuming plan is now plain text
        if not task_name or not plan_text:
            return jsonify({"success": False, "message": "Task name and plan text are required."}), 400
        
        success = save_user_task_structure(task_name, plan_text) # From vision.py
        if success:
            return jsonify({"success": True, "message": "Task structure saved successfully."})
        else:
            return jsonify({"success": False, "message": "Failed to save task structure."}), 500
    except Exception as e:
        logging.error(f"Error in /tasks/save: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/tasks/update', methods=['POST'])
def update_task_api():
    try:
        data = request.json
        task_name = data.get('name')  # Use name as identifier
        task_id = data.get('id')      # Also accept task ID
        new_plan_text = data.get('plan')
        
        if not new_plan_text:
            return jsonify({"success": False, "message": "New plan text is required."}), 400
            
        if not task_name and not task_id:
            return jsonify({"success": False, "message": "Either task name or task ID is required."}), 400

        # If task_id is provided, try to get the task name from the session
        if task_id: # Removed task_id in agent_state.task_sessions
            # This part needs adjustment if task_sessions is removed for ChromaDB
            loaded_session_for_name = agent_state._load_session_from_chromadb(task_id)
            if loaded_session_for_name: task_name = loaded_session_for_name.task_name
        if not task_name:
            return jsonify({"success": False, "message": "Could not determine task name for update."}), 400

        success = update_user_task_structure(task_name, new_plan_text)  # From vision.py
        if success:
            return jsonify({"success": True, "message": "Task structure updated successfully."})
        else:
            # update_user_task_structure might return False if task not found or DB error
            return jsonify({"success": False, "message": "Failed to update task structure (not found or DB error)."}), 404
    except Exception as e:
        logging.error(f"Error in /tasks/update: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/tasks/delete', methods=['POST'])
def delete_task_api():
    try:
        data = request.json
        task_name = data.get('name') # Use name as identifier
        if not task_name:
            return jsonify({"success": False, "message": "Task name is required."}), 400
        
        success = delete_user_task_structure(task_name) # From vision.py
        if success:
            return jsonify({"success": True, "message": "Task structure deleted successfully."})
        else:
            return jsonify({"success": False, "message": "Failed to delete task structure (not found or DB error)."}), 404 # Or 500
    except Exception as e:
        logging.error(f"Error in /tasks/delete: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/tasks/retrieve/<task_id_or_name>', methods=['GET']) # Can be ID or Name for flexibility
def retrieve_task_api(task_id_or_name: str):
    try:
        task_data = retrieve_user_task_structure(task_name=task_id_or_name) # Or task_id= if it's an ID
        if task_data:
            # Transform the data to match frontend expectations
            formatted_task = {
                "id": task_data["id"],
                "name": task_data["task_name"],
                "preview": task_data["plan_text"][:100] + "..." if len(task_data["plan_text"]) > 100 else task_data["plan_text"],
                "full_plan": task_data["plan_text"]
            }
            return jsonify({"success": True, "task": formatted_task})
        else:
            return jsonify({"success": False, "message": f"Task structure '{task_id_or_name}' not found."}), 404
    except Exception as e:
        logging.error(f"Error in /tasks/retrieve: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/tasks/history', methods=['GET'])
def get_all_task_sessions(): # Renamed to be more descriptive
    return jsonify(agent_state.get_task_history_list())

@app.route('/tasks/resume/<task_id>', methods=['POST'])
def resume_specific_task_route(task_id: str): 
    global agent_state
    
    
    if agent_state.current_task and agent_state.current_task.task_id != task_id and \
       agent_state.current_task.status == "active":
        logging.info(f"Switching task: Pausing current task {agent_state.current_task.task_id}")
        agent_state.pause_current_task()

    session_to_resume = agent_state.resume_task(task_id) 
    
    if session_to_resume:
        logging.info(f"API: Resuming task {task_id}. Set as current. UI state will update.")
        return jsonify({
            "status": "success",
            "message": f"Task '{session_to_resume.task_name}' resumed successfully.",
            **get_ui_update_state()
        })
    else:
        logging.warning(f"Failed to resume task {task_id}: Task not found or could not be resumed.")
        return jsonify({
            "status": "error",
            "message": f"Task {task_id} not found or could not be resumed.",
            **get_ui_update_state()
        }), 404

@app.route('/tasks/history/<task_id>', methods=['GET'])
def get_task_history(task_id: str):
    """Get the full history of a specific task."""
    session = agent_state._load_session_from_chromadb(task_id)
    if session:
        return jsonify(session.to_dict())
    logging.warning(f"Task history requested for non-existent task: {task_id}")
    return jsonify({"error": "Task not found"}), 404

@app.route('/tasks/thoughts/<task_id>', methods=['GET'])
def get_task_thoughts(task_id: str):
    """Get the agent thoughts for a specific task."""
    try:
        # With ChromaDB, we'd load the session first
        session = agent_state._load_session_from_chromadb(task_id)
        if session: return jsonify(session.agent_thoughts)
        return jsonify({"error": "Task not found"}), 404
    except Exception as e:
        logging.error(f"Error retrieving task thoughts for {task_id}: {e}", exc_info=True)
        return jsonify({"error": f"Error retrieving task thoughts: {str(e)}"}), 500

@app.route('/tasks/<task_id>', methods=['GET'])
def get_task_state_route(task_id):
    """Get the current state of a specific task."""
    try:
        # Load task from ChromaDB instead of task_sessions
        session = agent_state._load_session_from_chromadb(task_id)
        if session:
            return jsonify({**session.to_dict(), "ui_state": get_ui_update_state()})
        return jsonify({"error": "Task not found", "ui_state": get_ui_update_state()}), 404
    except Exception as e:
        logging.error(f"Error getting task state for {task_id}: {e}")
        return jsonify({
            "error": str(e),
            "ui_state": get_ui_update_state()
        }), 500

@app.route('/get_ui_state', methods=['GET'])
def get_ui_state_route():
    """Endpoint for frontend to poll for UI state updates."""
    return jsonify(get_ui_update_state())

# Helper functions defined at the end or imported, to be used by routes
def handle_task_completion(task: TaskSession, final_results: Optional[Dict] = None):
    """Handle task completion and update UI state."""
    print(f"\n[handle_task_completion] Task {task.task_id} completed with results:", final_results)
    
    # Log token usage found in search results
    if final_results and 'token_usage' in final_results:
        print(f"[handle_task_completion] Found token usage in search results: {final_results['token_usage']}")
        task._accumulate_tokens(final_results['token_usage'])
    
    task.status = "completed"
    task.end_time = datetime.now(timezone.utc)
    agent_state.status_message = f'Task completed: {task.task_name}'
    
    # Add a completion message to the conversation history
    task.conversation_history.append({
        "role": "system",
        "content": "Task completed successfully."
    })
    
    # Log the final token counts
    print(f"[handle_task_completion] Final token counts for task: {task.total_tokens}")
    
    # Save the final state
    agent_state._save_session_to_chromadb(task)
    
    # Update UI state
    ui_state = get_ui_update_state()
    print(f"[handle_task_completion] Final UI state token counts: {ui_state['totalTokens']}")
    return ui_state

def handle_task_error(session: TaskSession, error: Exception):
    logging.error(f"Task '{session.task_name}' (ID: {session.task_id}) encountered an error: {error}", exc_info=True)
    session.conversation_history.append({"role": "assistant", "content": f"An error occurred: {str(error)}. Task stopped."})
    session.status = "failed" # Mark as failed on error
    session.end_time = datetime.now(timezone.utc) # Use timezone.utc
    session.execution_log += f"\nERROR: {str(error)}"
    agent_state.is_task_running = False
    session.generator = None # Clear generator for this task
    agent_state.task_is_paused = False
    agent_state.status_message = f'Task failed: {session.task_name}'
    agent_state._save_session_to_chromadb(session) # Save changes

def generate_final_chat_response(session: TaskSession, final_results: Any):
    
    final_reasoning = "Task execution finished." 
    
    if isinstance(final_results, list) and final_results:
        last_result = final_results[-1]
        if isinstance(last_result, tuple) and len(last_result) >= 3:
            # last_result is (action_dict, success_bool, message_str, special_directive_dict_or_None)
            message = last_result[2] # message_str is the 3rd element
            
            assessment_match = re.search(r'Assessment:\s*(SUCCESS|FAILURE|RETRY_POSSIBLE)\s*-\s*(.*)', str(message))
            if assessment_match:
                status = assessment_match.group(1)
                reason = assessment_match.group(2)
                final_reasoning = f"Final Status: {status}. Reason: {reason}"
            else: 
                final_reasoning = f"Final message: {str(message)[:200]}" 
        elif isinstance(last_result, dict) and 'message' in last_result: 
            final_reasoning = f"Final message: {last_result['message']}"

    try:
        completion_message, chat_tokens = chat_with_user(
            session.conversation_history,
            session.task_name,
            final_results,
            final_reasoning
        )
        logging.info(f"[generate_final_chat_response] Received chat tokens: {chat_tokens}")
        session._accumulate_tokens(chat_tokens)
        session.conversation_history.append({"role": "assistant", "content": completion_message})
    except Exception as e:
        logging.error(f"[generate_final_chat_response] Error in chat_with_user during final response for task {session.task_id}: {e}", exc_info=True)
        session.conversation_history.append({"role": "assistant", "content": f"Task finished. Error generating summary: {str(e)}"})


def analyze_successful_steps(session: TaskSession, final_results: Any):
    
    try:
        task_was_successful = False
        if session.status == "completed": 
            
            if isinstance(final_results, list) and final_results:
                last_action_tuple = final_results[-1]
                # Each item in final_results is (action_dict, success_bool, message_str, special_directive_dict_or_None)
                if isinstance(last_action_tuple, tuple) and len(last_action_tuple) >=2: # Check for at least 2 elements
                    task_was_successful = last_action_tuple[1] # success is the second element
                    # Further check if the action itself was task_complete
                    if isinstance(last_action_tuple[0], dict) and last_action_tuple[0].get('action_type') == 'task_complete':
                        task_was_successful = True 

        if task_was_successful:
            successful_steps_for_analysis = [
                res for res in final_results
                # Each res is (action_dict, success_bool, message_str, special_directive_dict_or_None)
                if isinstance(res, tuple) and len(res) >= 2 and res[1] and # res[1] is success_bool
                isinstance(res[0], dict) and res[0].get('action_type') not in ['STOP', 'REPLAN', 'CRITIQUE_FAILED', 'ask_user', 'INFORM_USER', 'task_complete']
            ]

            if successful_steps_for_analysis:
                logging.info(f"[analyze_successful_steps] Analyzing {len(successful_steps_for_analysis)} successful steps for reinforcements for task {session.task_id}.")
                learnings, reinforcement_tokens = analyze_feedback_and_generate_reinforcements(
                    original_instruction=session.task_name,
                    execution_results=successful_steps_for_analysis,
                    user_feedback="Task completed successfully by agent.",
                    llm_model=model
                )
                logging.info(f"[analyze_successful_steps] Received reinforcement tokens: {reinforcement_tokens}")
                session._accumulate_tokens(reinforcement_tokens)
            else:
                logging.info(f"[analyze_successful_steps] No specific successful steps found to analyze for reinforcements for task {session.task_id}, though task was successful.")
        else:
            logging.info(f"[analyze_successful_steps] Task {session.task_id} not marked as successful for reinforcement analysis (status: {session.status}).")

    except Exception as e:
        logging.error(f"[analyze_successful_steps] Error analyzing successful steps for reinforcements (task {session.task_id}): {e}", exc_info=True)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part in request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file.'}), 400
    from utils.sanitize_util import sanitize_filename
    filename = sanitize_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    return jsonify({'success': True, 'filename': filename, 'filepath': save_path})

def start_frontend_dev_server():
    # Determine the absolute path to the directory containing app_flask.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the frontend directory relative to the script's directory
    frontend_dir = os.path.join(script_dir, "frontend")
    if not os.path.isdir(frontend_dir):
        logging.error(f"Frontend directory not found: {frontend_dir}")
        return

    command = []
    if sys.platform == "win32":
        command = ['npm.cmd', 'start'] # Or just 'npm' if it's in PATH and resolves correctly
    else:
        command = ['npm', 'start']

    logging.info(f"Attempting to start frontend dev server in {frontend_dir} with command: {' '.join(command)}")
    try:
        # Use shell=True on Windows if npm.cmd needs it, otherwise shell=False is safer
        process = subprocess.Popen(command, cwd=frontend_dir, shell=(sys.platform == "win32"))
        logging.info(f"Frontend dev server process started (PID: {process.pid}). Output will be in its own console or logs.")
    except FileNotFoundError:
        logging.error(f"'npm' command not found. Please ensure Node.js and npm are installed and in your system's PATH.")
    except Exception as e:
        logging.error(f"Failed to start frontend dev server: {e}", exc_info=True)

@app.route('/update_api_key', methods=['POST'])
def update_api_key():
    try:
        data = request.get_json()
        if not data or 'api_key' not in data:
            return jsonify({
                'success': False,
                'message': 'API key is required'
            }), 400

        new_api_key = data['api_key']
        logging.info("Received API key update request")
        
        # Validate the API key by attempting to configure Gemini API
        try:
            import google.generativeai as genai
            from google import genai as genait
            logging.info("Configuring genai with new key")
            genai.configure(api_key=new_api_key)
            
            logging.info("Creating test client")
            test_client = genait.Client(api_key=new_api_key)
            
            logging.info("Creating test model")
            test_model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-04-17")
            
            # Test the API key with a simple generation
            logging.info("Testing model with simple generation")
            response = test_model.generate_content("Test")
            if not response or not response.candidates:
                logging.error("API key validation failed - no response generated")
                return jsonify({
                    'success': False,
                    'message': 'API key validation failed: Could not generate test response'
                }), 400
                
        except Exception as e:
            logging.error(f"API key validation failed: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'message': f'Invalid API key: {str(e)}'
            }), 400
        
        # If validation passes, update the config file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
        logging.info(f"Updating config file at {config_path}")
        
        with open(config_path, 'r') as f:
            content = f.read()
            
        # Update the API key in the config file
        new_content = re.sub(
            r'API_KEY\s*=\s*["\'].*?["\']',
            f'API_KEY = "{new_api_key}"',
            content
        )
        
        with open(config_path, 'w') as f:
            f.write(new_content)
            
        # Update the global model in config.py
        from config import configure_gemini_api
        if not configure_gemini_api(new_api_key):
            return jsonify({
                'success': False,
                'message': 'Failed to configure Gemini API with new key'
            }), 500
            
        # Update the ui_agent's model
        global ui_agent
        from config import model
        ui_agent = UIAgent(model)
        
        return jsonify({
            'success': True,
            'message': 'API key updated successfully'
        })
        
    except Exception as e:
        logging.error(f"Error updating API key: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error updating API key: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Launching Flask UI...")
    try:
        # Check if the log file exists and is writable
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as f: # Append mode
                f.write(f"\n--- Flask App Started: {datetime.now(timezone.utc).isoformat()} ---\n") # Use timezone.utc
        except Exception as e:
            print(f"Warning: Could not write to log file {LOG_FILE}: {e}")

        # Start the frontend development server in a separate thread
        logging.info("Preparing to start frontend development server...")
        frontend_thread = threading.Thread(target=start_frontend_dev_server, daemon=True)
        frontend_thread.start()

        # Use a more robust server configuration
        app.run(
            host="0.0.0.0",
            port=5001, # Default port
            debug=True, # Enable debug mode for development
            use_reloader=False,  # Disable reloader if it causes issues, especially with background threads/processes
            threaded=True  # Enable threading for better handling of concurrent requests if your agent does blocking IO
        )
    except OSError as e:
        if "address already in use" in str(e).lower():
            print(f"Port 5001 is in use. Trying alternative port 5002...")
            try:
                app.run(host="0.0.0.0", port=5002, debug=True, use_reloader=False, threaded=True)
            except OSError as e2:
                print(f"Failed to start server on alternative port 5002: {e2}")
                sys.exit(1)
        else:
            print(f"OSError starting server: {e}")
            sys.exit(1)
    except Exception as e: # Catch any other exception during app.run
        print(f"Unexpected error starting Flask server: {e}")
        sys.exit(1)
