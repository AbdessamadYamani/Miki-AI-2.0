import logging
import hashlib
import time
from typing import List, Dict, Optional
import re,sys,os
import json
from typing import Tuple
from tools.token_usage_tool import _get_token_usage
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    chroma_client, user_task_structures_collection, BLUE, RESET, RED, task_executions_collection, GREEN,
    get_model
)



def save_user_task_structure(task_name: str, plan_text_string: str) -> bool:
    """Saves a user-defined task structure (plan) to ChromaDB."""
    if not chroma_client or not user_task_structures_collection:
        logging.error("ChromaDB not available. Cannot save user task structure.")
        return False
    if not task_name or not plan_text_string:
        logging.error("Task name or plan text string is empty. Cannot save.")
        return False

    try:
        # Use a hash of the task name as the ID for uniqueness and easy retrieval/update
        task_id = hashlib.md5(task_name.encode('utf-8')).hexdigest()

        # For saving, we directly add/update. If it exists, ChromaDB's add with the same ID acts as an upsert.
        # However, the app.py logic for "Save New" vs "Update" handles the user intent.
        # Here, we just perform the upsert.
        # If you strictly want 'save' to fail if exists, you'd do a .get() first.
        # But for simplicity and matching ChromaDB's behavior, 'add' with ID is fine.

        metadata = {
            "task_name": task_name,
            "plan_text": plan_text_string, # Store the plain text string
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # The document content is what ChromaDB uses for embedding and similarity search.
        # Using the task name itself is a good starting point.
        user_task_structures_collection.add(
            documents=[task_name],
            metadatas=[metadata],
            ids=[task_id]
        )
        logging.info(f"Saved/Updated user task structure to ChromaDB (ID: {task_id}) for task: '{task_name}'")
        print(f"{BLUE}[ChromaDB] Successfully saved/updated user task structure: '{task_name}'{RESET}")
        return True
    except Exception as e:
        logging.error(f"Error saving/updating user task structure to ChromaDB: {e}", exc_info=True)
        print(f"{BLUE}[ChromaDB] {RED}FAILED{BLUE} to save/update user task structure: {e}{RESET}")
        return False


def load_user_task_structures() -> List[Dict]:
    """Loads all user-defined task structures from ChromaDB."""
    if not chroma_client or not user_task_structures_collection:
        logging.error("ChromaDB not available. Cannot load user task structures.")
        return []
    try:
        results = user_task_structures_collection.get(include=['metadatas']) # Remove 'ids' from include
        loaded_tasks = []
        if results and results['ids']:
            for i in range(len(results['ids'])):
                task_data = {
                    "id": results['ids'][i], # Get ID from the results
                    "task_name": results['metadatas'][i].get('task_name', 'Unnamed Task'),
                    "plan_text": results['metadatas'][i].get('plan_text', ''), # Changed from plan_json
                    "timestamp": results['metadatas'][i].get('timestamp', 'N/A')
                }
                loaded_tasks.append(task_data)
        logging.info(f"Loaded {len(loaded_tasks)} user task structures from ChromaDB.")
        return loaded_tasks
    except Exception as e:
        logging.error(f"Error loading user task structures from ChromaDB: {e}", exc_info=True)
        print(f"{BLUE}[ChromaDB] {RED}FAILED{BLUE} to load user task structures: {e}{RESET}")
        return []


def retrieve_user_task_structure(task_name: Optional[str] = None, task_id: Optional[str] = None) -> Optional[Dict]:
    """Retrieves a single user task structure by name or ID."""
    if not chroma_client or not user_task_structures_collection:
        logging.error("ChromaDB not available. Cannot retrieve user task structure.")
        return None
    if not task_name and not task_id:
        logging.error("Must provide either task_name or task_id to retrieve user task structure.")
        return None

    try:
        query_id = task_id if task_id else hashlib.md5(task_name.encode('utf-8')).hexdigest()
        results = user_task_structures_collection.get(ids=[query_id], include=['metadatas'])
        if results and results['ids']:
            task_data = {
                "id": results['ids'][0],
                "task_name": results['metadatas'][0].get('task_name', 'Unnamed Task'),
                "plan_text": results['metadatas'][0].get('plan_text', ''), # Changed from plan_json
                "timestamp": results['metadatas'][0].get('timestamp', 'N/A')
            }
            logging.info(f"Retrieved user task structure (ID: {query_id}).")
            return task_data
        else:
            logging.info(f"User task structure not found (ID/Name: {query_id if task_id else task_name}).")
            return None
    except Exception as e:
        query_identifier = task_id if task_id else task_name
        logging.error(f"Error retrieving user task structure from ChromaDB (Query Identifier: {query_identifier}): {e}", exc_info=True)
        print(f"{BLUE}[ChromaDB] {RED}FAILED{BLUE} to retrieve user task structure: {e}{RESET}")
        return None


def update_user_task_structure(task_name: str, new_plan_text_string: str) -> bool: # Renamed parameter
    """Updates an existing user-defined task structure's plan_text in ChromaDB using the collection's update method."""
    if not chroma_client or not user_task_structures_collection:
        logging.error("ChromaDB not available. Cannot update user task structure.")
        return False
    if not task_name:
        logging.error("Task name is empty. Cannot update.")
        return False

    task_id = hashlib.md5(task_name.encode('utf-8')).hexdigest()

    # Verify the task exists before attempting to update
    # retrieve_user_task_structure uses .get() which is suitable here.
    existing_task_data = retrieve_user_task_structure(task_id=task_id)
    if not existing_task_data:
        logging.warning(f"User task structure '{task_name}' (ID: {task_id}) not found for update.")
        print(f"{BLUE}[ChromaDB] Task structure '{task_name}' not found for update.{RESET}")
        return False
    
    try:
        new_metadata = {
            "task_name": task_name,  # Keep task_name consistent in metadata
            "plan_text": new_plan_text_string, # The new plan text
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S") # Update timestamp
        }

        user_task_structures_collection.update(
            ids=[task_id],
            metadatas=[new_metadata]
            # We are only updating metadata, so no need to provide 'documents' or 'embeddings'
            # unless the task_name (which is the document) itself was also changed and needs re-embedding.
        )
        logging.info(f"Updated user task structure in ChromaDB (ID: {task_id}) for task: '{task_name}'")
        print(f"{BLUE}[ChromaDB] Successfully updated user task structure: '{task_name}'{RESET}")
        return True
    except Exception as e:
        logging.error(f"Error updating user task structure in ChromaDB (ID: {task_id}): {e}", exc_info=True)
        print(f"{BLUE}[ChromaDB] {RED}FAILED{BLUE} to update user task structure for '{task_name}': {e}{RESET}")
        return False

def delete_user_task_structure(task_name: str) -> bool:
    """Deletes a user-defined task structure from ChromaDB by name."""
    if not chroma_client or not user_task_structures_collection:
        logging.error("ChromaDB not available. Cannot delete user task structure.")
        return False
    if not task_name:
        logging.error("Task name is empty. Cannot delete.")
        return False
    try:
        task_id = hashlib.md5(task_name.encode('utf-8')).hexdigest()
        # Check if it exists before deleting to provide better feedback
        existing = user_task_structures_collection.get(ids=[task_id])
        if not existing or not existing['ids']:
            logging.warning(f"User task structure '{task_name}' (ID: {task_id}) not found for deletion.")
            print(f"{BLUE}[ChromaDB] User task structure '{task_name}' not found for deletion.{RESET}")
            return False # Or True if "not found" is an acceptable outcome for delete

        user_task_structures_collection.delete(ids=[task_id])
        logging.info(f"Deleted user task structure from ChromaDB (ID: {task_id}) for task: '{task_name}'")
        print(f"{BLUE}[ChromaDB] Successfully deleted user task structure: '{task_name}'{RESET}")
        return True
    except Exception as e:
        logging.error(f"Error deleting user task structure from ChromaDB: {e}", exc_info=True)
        print(f"{BLUE}[ChromaDB] {RED}FAILED{BLUE} to delete user task structure: {e}{RESET}")
        return False
def find_similar_user_task_structure(query_instruction: str, n_results: int = 1) -> Optional[Dict]:
    """Finds the most similar user-defined task structure based on the query instruction."""
    if not chroma_client or not user_task_structures_collection:
        logging.error("ChromaDB not available. Cannot search for similar user task structures.")
        return None
    if not query_instruction: return None
    try:
        count = user_task_structures_collection.count()
        if count == 0:
            logging.info("No user task structures in DB to search.")
            return None

        # Query using the instruction. The document content is the task name.
        # We are searching for task names similar to the user's instruction.
        results = user_task_structures_collection.query(
            query_texts=[query_instruction],
            n_results=min(n_results, count), # Ensure n_results <= collection count
            include=['metadatas', 'distances'] # Include distances. IDs are typically returned by default.
        )

        if results and results['ids'] and results['ids'][0] and results['distances'] and results['distances'][0]:
            # ChromaDB returns lists of lists for these, even for n_results=1
            best_match_id = results['ids'][0][0]
            best_match_distance = results['distances'][0][0]
            best_match_metadata = results['metadatas'][0][0]

            # Cosine distance: 0 is identical, higher is less similar.
            # Sentence Transformers typically output distances in [0, 2], often [0, 1] for similar items.
            # A threshold like 0.5 or 0.6 might be reasonable for "similar enough".
            # This needs tuning based on the embedding model and desired sensitivity.
            similarity_threshold = 0.6 # Example threshold, adjust as needed

            if best_match_distance < similarity_threshold:
                logging.info(f"Found similar user task structure (ID: {best_match_id}, Name: '{best_match_metadata.get('task_name')}', Distance: {best_match_distance:.4f}) for query: '{query_instruction[:70]}...'")
                # Construct the full task data to return
                task_data = {
                    "id": best_match_id,
                    "task_name": best_match_metadata.get('task_name', 'Unnamed Task'),
                    "plan_text": best_match_metadata.get('plan_text', ''), # Changed from plan_json
                    "timestamp": best_match_metadata.get('timestamp', 'N/A')
                }
                return task_data
            else:
                logging.info(f"Most similar user task structure (Name: '{best_match_metadata.get('task_name')}', Distance: {best_match_distance:.4f}) is below similarity threshold ({similarity_threshold}).")
                return None
        else:
            logging.info(f"Query: '{query_instruction[:70]}...' - No similar user task structures retrieved or result format unexpected.")
            return None
    except Exception as e:
        logging.error(f"Error searching for similar user task structures: {e}", exc_info=True)
        print(f"{BLUE}[ChromaDB] Query: '{query_instruction[:70]}...' - {RED}FAILED{BLUE} to search for similar user task structures: {e}{RESET}")
        return None


def _analyze_and_categorize_task(
    instruction: str,
    llm_model: genai.GenerativeModel
) -> Tuple[Optional[str], List[str], str, Dict[str, int]]:
    """
    Analyzes the original instruction to categorize the task and break it down into sub-tasks if complex.
    Returns: (category, sub_tasks_list, reasoning, token_usage)
    """
    # Check if llm_model is None and try to get a new instance
    if llm_model is None:
        llm_model = get_model()
        if llm_model is None:
            logging.error("LLM model is None and could not be initialized")
            return None, [], "Failed to initialize LLM model", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    logging.info(f"Analyzing and categorizing task: {instruction}")
    prompt = f"""
You are a task analysis expert. Your goal is to understand a user's request, categorize its complexity,
and if it's medium or hard, break it down into a logical sequence of high-level sub-tasks.

User's Request: "{instruction}"

Analysis Instructions:
1.  Determine the overall complexity of the request. Categorize it as 'simple', 'medium', or 'hard'.
    -   'simple': Can likely be done in 1-3 direct actions (e.g., "open notepad", "what's the weather?").
    -   'medium': Requires a few coordinated steps or a single script (e.g., "download a file and then extract it to a folder", "write a short python script to rename files").
    -   'hard': Involves multiple distinct phases, significant code generation, or complex interactions (e.g., "build a small web app", "create a data analysis pipeline and generate a report").
2.  If the category is 'medium' or 'hard', provide a list of 2-5 high-level sub-tasks. Each sub-task should be a clear, actionable step.
    If 'simple', the sub-tasks list can be empty or contain just the original instruction.
3.  Provide a brief reasoning for your categorization and sub-task breakdown.

Output Format (MUST be a single valid JSON object ONLY):
{{
  "category": "simple|medium|hard",
  "sub_tasks": ["Sub-task 1 description", "Sub-task 2 description", ...],
  "reasoning": "Your reasoning here."
}}

Example for a 'hard' task "Create a python script to fetch weather data and save it to csv":
{{
  "category": "hard",
  "sub_tasks": [
    "Design the script: identify necessary libraries and steps (API key, request, parsing, CSV writing).",
    "Write the Python script code to fetch weather data from an API.",
    "Implement error handling and data parsing in the script.",
    "Add functionality to save the parsed data to a CSV file.",
    "Test the script with a sample city."
  ],
  "reasoning": "This is a hard task as it involves multiple coding steps, API interaction, and file I/O, requiring careful planning and implementation."
}}
"""
    try:
        response = llm_model.generate_content(prompt) # Assuming safety_settings are default or handled by model instance
        token_usage = _get_token_usage(response)
        # Ensure response_text is correctly accessed
        raw_response_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            raw_response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()

        if not raw_response_text:
            block_reason_str = "Unknown (No text part in response or empty text)"
            finish_reason_str = "Unknown"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                block_reason_str = str(response.prompt_feedback.block_reason)
            if response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                finish_reason_str = str(response.candidates[0].finish_reason)
                if finish_reason_str == "SAFETY" and "SAFETY" not in block_reason_str: # Check if block_reason is already set
                    block_reason_str = "SAFETY (from finish_reason)"
                elif finish_reason_str != "STOP" and block_reason_str.startswith("Unknown"):
                     block_reason_str = f"Non-STOP finish_reason: {finish_reason_str}"

            logging.error(f"Task analysis LLM returned empty or blocked response. Block Reason: {block_reason_str}, Finish Reason: {finish_reason_str}")
            return None, [], f"LLM returned empty or blocked response for task analysis (Block: {block_reason_str}, Finish: {finish_reason_str}).", token_usage

        # Log the raw response before attempting to parse, for easier debugging
        logging.debug(f"Raw response text for task analysis before JSON parsing: '{raw_response_text}'")

        # Clean potential markdown fences from the response
        cleaned_response_text = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', raw_response_text, flags=re.MULTILINE | re.IGNORECASE).strip()
        logging.debug(f"Cleaned response text for task analysis: '{cleaned_response_text}'")

        # Attempt to parse the JSON
        analysis_data = json.loads(cleaned_response_text)
        category = analysis_data.get("category")
        sub_tasks = analysis_data.get("sub_tasks", [])
        reasoning = analysis_data.get("reasoning")
        logging.info(f"Task Analysis Complete. Category: {category}, Sub-tasks: {len(sub_tasks)}. Reasoning: {reasoning}")
        return category, sub_tasks, reasoning, token_usage
    except json.JSONDecodeError as e:
        # This block should now be hit, and its logs should appear.
        logging.error(f"Specific Handler: Error decoding JSON from task analysis LLM: {e}", exc_info=True)
        problem_text_log = "<raw_response_text not available or error during repr>"
        if 'raw_response_text' in locals():
            try:
                # Use repr() for safer logging of potentially problematic strings
                problem_text_log = repr(cleaned_response_text if 'cleaned_response_text' in locals() and cleaned_response_text else raw_response_text)
            except Exception as repr_err:
                problem_text_log = f"<Error getting repr of raw_response_text: {repr_err}>"
        
        logging.error(f"Specific Handler: Problematic response text for task analysis (repr): {problem_text_log}")
        return None, [], f"Error decoding JSON from task analysis LLM: {e}", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    except Exception as e:
        logging.error(f"Error during task analysis: {e}", exc_info=True)
        return None, [], f"Error during task analysis: {e}", {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}


def save_task_execution_to_db(
    original_instruction: str,
    execution_summary_doc: str, # This will be the document for ChromaDB
    full_results_json: str, # JSON string of the results list
    status: str, # "success" or "failure" or "incomplete"
    plan_source: str,
    user_feedback: Optional[str] = None
) -> bool:
    if not chroma_client or not task_executions_collection:
        logging.error("ChromaDB not available. Cannot save task execution.")
        return False
    try:
        # Create a unique ID, e.g., based on instruction and timestamp
        execution_id = hashlib.md5(f"{original_instruction}{time.time()}".encode('utf-8')).hexdigest()
        metadata = {
            "original_instruction": original_instruction,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "plan_source": plan_source,
            "full_results_json": full_results_json # Store detailed results as JSON string
        }
        if user_feedback:
            metadata["user_feedback"] = user_feedback

        task_executions_collection.add(
            documents=[execution_summary_doc],
            metadatas=[metadata],
            ids=[execution_id]
        )
        logging.info(f"Saved task execution to ChromaDB (ID: {execution_id}, Status: {status}) for: '{original_instruction[:100]}...'")
        print(f"{BLUE}[ChromaDB] Successfully saved task execution (Status: {status}) for: '{original_instruction[:50]}...'{RESET}")
        return True
    except Exception as e:
        logging.error(f"Error saving task execution to ChromaDB: {e}", exc_info=True)
        print(f"{BLUE}[ChromaDB] {RED}FAILED{BLUE} to save task execution: {e}{RESET}")
        return False

def retrieve_similar_task_executions_from_db(query_instruction: str, n_results: int = 3) -> List[Dict]:
    """Retrieves similar task executions from ChromaDB."""
    if not chroma_client or not task_executions_collection:
        logging.error("ChromaDB not available. Cannot retrieve task executions.")
        return []
    if not query_instruction: return []
    try:
        count = task_executions_collection.count()
        if count == 0: return []
        results = task_executions_collection.query(query_texts=[query_instruction], n_results=min(n_results, count))
        retrieved_tasks = [{"document": doc, "metadata": meta} for doc, meta in zip(results['documents'][0], results['metadatas'][0])] if results and results['documents'] else []
        if retrieved_tasks:
            print(f"{BLUE}[ChromaDB] Query: '{query_instruction[:70]}...' - Retrieved {GREEN}{len(retrieved_tasks)}{BLUE} similar task(s). First summary: '{retrieved_tasks[0]['document'][:70]}...'{RESET}")
        else:
            print(f"{BLUE}[ChromaDB] Query: '{query_instruction[:70]}...' - {RED}No similar task executions retrieved.{RESET}")
        return retrieved_tasks
    except Exception as e:
        logging.error(f"Error retrieving task executions from ChromaDB: {e}", exc_info=True)
        print(f"{BLUE}[ChromaDB] Query: '{query_instruction[:70]}...' - {RED}FAILED{BLUE} to retrieve similar task executions: {e}{RESET}")
        return []

