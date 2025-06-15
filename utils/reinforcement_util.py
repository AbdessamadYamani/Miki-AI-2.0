import os,sys
import json
import logging
import hashlib
import time
from typing import List
import google.generativeai as genai
import re
from typing import List,Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import  chroma_client, reinforcements_collection,BLUE, RESET, RED, GREEN
def save_reinforcement_to_db(reinforcement_text: str, original_instruction: str, source_type: str = "llm_generated") -> bool:
    """Saves a single reinforcement/learning to ChromaDB, avoiding duplicates based on text."""
    if not chroma_client or not reinforcements_collection:
        logging.error("ChromaDB not available. Cannot save reinforcement.")
        return False
    try:
        reinforcement_id = hashlib.md5(reinforcement_text.encode('utf-8')).hexdigest()
        existing_by_id = reinforcements_collection.get(ids=[reinforcement_id])
        if existing_by_id and existing_by_id['ids']:
            logging.info(f"Reinforcement already exists (ID: {reinforcement_id}): '{reinforcement_text[:100]}...'")
            return True
        # No blue note here as it's not a new save

        reinforcements_collection.add(
            documents=[reinforcement_text],
            metadatas=[{
                "original_instruction": original_instruction,
                "source_type": source_type,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }],
            ids=[reinforcement_id]
        )
        logging.info(f"Saved reinforcement to ChromaDB (ID: {reinforcement_id}): '{reinforcement_text[:100]}...'")
        print(f"{BLUE}[ChromaDB] Successfully saved reinforcement: '{reinforcement_text[:50]}...'{RESET}")
        return True
    except Exception as e:
        logging.error(f"Error saving reinforcement to ChromaDB: {e}", exc_info=True)
        print(f"{BLUE}[ChromaDB] {RED}FAILED{BLUE} to save reinforcement: {e}{RESET}")
        return False

def retrieve_relevant_reinforcements_from_db(query_text: str, n_results: int = 5) -> List[str]:
    """Retrieves relevant reinforcements from ChromaDB using semantic search."""
    if not chroma_client or not reinforcements_collection:
        logging.error("ChromaDB not available. Cannot retrieve reinforcements.")
        return []
    if not query_text: return []
    try:
        # Ensure n_results is not greater than the number of items in the collection
        count = reinforcements_collection.count()
        if count == 0: return []
        actual_n_results = min(n_results, count)

        results = reinforcements_collection.query(
            query_texts=[query_text],
            n_results=actual_n_results
        )
        retrieved_docs = results['documents'][0] if results and results['documents'] else []
        if retrieved_docs:
            print(f"{BLUE}[ChromaDB] Query: '{query_text[:70]}...' - Retrieved {GREEN}{len(retrieved_docs)}{BLUE} reinforcement(s). First: '{retrieved_docs[0][:70]}...'{RESET}")
        else:
            print(f"{BLUE}[ChromaDB] Query: '{query_text[:70]}...' - {RED}No reinforcements retrieved.{RESET}")
        return retrieved_docs
    except Exception as e:
        logging.error(f"Error retrieving reinforcements from ChromaDB: {e}", exc_info=True)
        print(f"{BLUE}[ChromaDB] Query: '{query_text[:70]}...' - {RED}FAILED{BLUE} to retrieve reinforcements: {e}{RESET}")
        return []

def analyze_feedback_and_generate_reinforcements(
    original_instruction: str,
    execution_results: List[Tuple[dict, bool, str]],
    user_feedback: str,
    llm_model: genai.GenerativeModel
) -> List[str]:
    logging.info("Analyzing feedback and execution for reinforcements...")

    if not execution_results:
        logging.warning("Cannot generate reinforcements: No execution results provided.")
        return []

    summary = "\nExecution Summary:\n"
    for i, (action, success, message) in enumerate(execution_results):
        status = "[OK]" if success else "[ERROR]"
        action_type = action.get('action_type', 'Unknown')

        reasoning_part = ""
        if "| Assessment: SUCCESS - " in message:
            reasoning_part = message.split("| Assessment: SUCCESS - ", 1)[-1]
        elif "| Assessment: FAILURE - " in message:
            reasoning_part = message.split("| Assessment: FAILURE - ", 1)[-1]
        elif "| Assessment: RETRY_POSSIBLE - " in message:
            reasoning_part = message.split("| Assessment: RETRY_POSSIBLE - ", 1)[-1]
        else:
             reasoning_part = message.split("| Assessment:", 1)[0].strip()

        summary += f"Step {i+1} ({action_type}): {status} - {reasoning_part}\n"

    feedback_section = f"User Feedback: {user_feedback}" if user_feedback else "User Feedback: (No feedback provided)"

    prompt = f"""
Respond only with a single valid JSON array. Do not include any markdown fences, explanatory text, leading/trailing whitespace, or other characters.
You are a Reinforcement Learning Analyst for a PC automation agent.
Your goal is to extract concise, actionable learnings ("reinforcements") from a successfully completed task execution log and optional user feedback. These learnings will help the agent perform similar tasks better or faster in the future.

Original User Instruction: "{original_instruction}"

Execution Summary (Successful Steps):
{summary}

{feedback_section}

**Analysis Task:**
Based *only* on the successful execution summary and user feedback:
1.  Identify key insights about efficient methods used (e.g., specific shortcuts, effective commands).
2.  Note any potential improvements suggested by the feedback (e.g., "Could have used Alt+Tab instead of clicking").
3.  Extract generalizable rules or heuristics (e.g., "Waiting 1s after opening Notepad prevents typing errors", "Using 'run_shell_command' is faster for creating empty files than GUI").
4.  Focus on what worked well or could be improved for *similar future tasks*.

**Output Format:**
Generate a JSON list of strings. Each string should be a concise learning/reinforcement (max 1-2 sentences).

Example Output:
[
  "Using 'Ctrl+S' is a reliable way to save files in most applications.",
  "Waiting 1 second after opening an application can prevent errors when typing immediately.",
  "For creating empty files, 'run_shell_command' with 'echo.' is faster than using the GUI.",
  "If a click fails, check if the window needs to be focused first using 'focus_window'."
]

CRITICAL: Respond ONLY with the raw JSON list, starting with `[` and ending with `]`. Do NOT include ```json or any other text.
"""

    try:
        safety_settings = {}
        response = llm_model.generate_content(prompt, safety_settings=safety_settings)
        raw_response_text = getattr(response, 'text', '').strip()

        if not raw_response_text:
             if response.prompt_feedback.block_reason: logging.error(f"Reinforcement analysis blocked: {response.prompt_feedback.block_reason}")
             else: logging.error("Reinforcement analysis LLM returned empty response.")
             return []

        logging.debug(f"LLM raw response for reinforcements:\n{raw_response_text}")

        learnings = []
        try:
            cleaned_text = re.sub(r'^[^{\[]*', '', raw_response_text, flags=re.DOTALL).strip()
            cleaned_text = re.sub(r'[^}\]]*$', '', cleaned_text, flags=re.DOTALL).strip()
            if not cleaned_text:
                logging.error("Response was empty after aggressive cleaning for reinforcements.")
                return []
            else:
                learnings = json.loads(cleaned_text)
                logging.info("JSON parsing successful for reinforcements after aggressive cleaning.")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode reinforcements JSON after aggressive cleaning: {e}\nCleaned text: '{cleaned_text if 'cleaned_text' in locals() else 'N/A'}'")
            return []

        saved_count = 0
        if isinstance(learnings, list) and all(isinstance(item, str) for item in learnings):
            logging.info(f"LLM generated {len(learnings)} potential new reinforcements.")
            for learning_text in learnings:
                if save_reinforcement_to_db(learning_text, original_instruction, source_type="llm_generated_from_feedback"):
                    saved_count += 1
            logging.info(f"Successfully saved {saved_count}/{len(learnings)} new reinforcements to ChromaDB.")
            return learnings 
        else:
            logging.error(f"Reinforcement analysis response was not a list of strings: {learnings}")
            return []

    except ValueError as ve:
         logging.error(f"ValueError accessing reinforcement LLM: {ve}")
         return []
    except Exception as e:
        logging.error(f"Error generating reinforcements: {e}", exc_info=True)
        return []
