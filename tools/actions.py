from typing import List, Dict, Optional, Tuple, Union, Any
import os
import logging,re
import google.generativeai as genai
from PIL import  Image
from tools.token_usage_tool import _get_token_usage
from vision.vis import image_to_base64,_hash_pil_image,capture_full_screen
from agents.ai_agent import UIAgent
from tools.web_search_tool import search_web_for_info
import time
import sys
import pyautogui
import pyperclip
from vision.vis import focus_window_by_title,get_screen_description_from_gemini
from tools.web_search_tool import search_web_for_info,navigate_web
from tools.youtube_tool import process_and_store_youtube_videos, search_youtube_transcripts
from utils.file_util import _execute_read_file
import subprocess
from tools.image_generating import generate_or_edit_image
import mimetypes
import pathlib
import httpx
import tempfile
from tools.files_upload import process_files_from_urls,process_local_files

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from config import CONFIRM_RISKY_COMMANDS,DEBUG_DIR,get_model as model

# Supported file types for Gemini API (moved from test_files.py)
SUPPORTED_MIME_TYPES = [
    'application/pdf',
    'application/x-javascript', 'text/javascript',
    'application/x-python', 'text/x-python', # type: ignore
    'text/plain', # type: ignore
    'text/html', # type: ignore
    'text/css', # type: ignore
    'text/md', # type: ignore
    'text/csv', # type: ignore
    'text/xml', # type: ignore
    'text/rtf' # type: ignore
]

ZERO_TOKEN_USAGE = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

def get_mime_type_from_path(filepath: str) -> Optional[str]:
    mime_type, _ = mimetypes.guess_type(str(filepath))
    return mime_type

def get_mime_type_from_url(url: str) -> Optional[str]:
    mime_type, _ = mimetypes.guess_type(url)
    if not mime_type:
        try:
            response = httpx.head(url, follow_redirects=True)
            response.raise_for_status() # Raise an exception for bad status codes
            mime_type = response.headers.get('content-type', '').split(';')[0]
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error getting MIME type for {url}: {e}")
            mime_type = None
        except Exception as e:
            logging.error(f"Error getting MIME type for {url}: {e}")
            mime_type = None
    return mime_type

def _execute_write_file(file_path: str, content: str, append: bool = False) -> Tuple[bool, str]:
    """
    Writes or appends string content to a specified file.
    Creates parent directories if they don't exist.

    Args:
        file_path: The full path to the target file.
        content: The string content to write.
        append: If True, append to the file. If False (default), overwrite the file.

    Returns:
        A tuple (success: bool, message: str).
    """
    action_desc = "append to" if append else "write to"
    logging.info(f"Attempting to {action_desc} file: {file_path}")
    try:

        directory = os.path.dirname(file_path)

        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")
            except OSError as e:

                logging.error(f"Error creating directory '{directory}': {e}")
                return False, f"Error creating directory '{directory}': {e}"


        mode = 'a' if append else 'w'


        with open(file_path, mode, encoding='utf-8') as f:
            bytes_written = f.write(content)

            if append and not content.endswith('\n'):
                 f.write('\n')
                 bytes_written += 1


        logging.info(f"Successfully {action_desc} file: {file_path} ({bytes_written} bytes written)")
        return True, f"Successfully {action_desc} file: {file_path}"

    except FileNotFoundError:

        logging.error(f"Error {action_desc} file: Path not found or invalid: {file_path}")
        return False, f"Error {action_desc} file: Path not found or invalid: {file_path}"
    except PermissionError:
        logging.error(f"Error {action_desc} file: Permission denied for path: {file_path}")
        return False, f"Error {action_desc} file: Permission denied for path: {file_path}"
    except IsADirectoryError:
         logging.error(f"Error {action_desc} file: Specified path is a directory: {file_path}")
         return False, f"Error {action_desc} file: Specified path is a directory: {file_path}"
    except OSError as e:

        logging.error(f"Error {action_desc} file: OS error for path '{file_path}': {e}")
        return False, f"Error {action_desc} file: OS error for path '{file_path}': {e}"
    except Exception as e:

        logging.error(f"Unexpected error {action_desc} file '{file_path}': {e}", exc_info=True)
        return False, f"Unexpected error {action_desc} file '{file_path}': {e}"


def execute_shell_command(command: str, working_directory: Optional[str] = None, allow_confirmation_prompt: bool = True) -> Union[Tuple[int, str], Tuple[int, str, Dict[str, Any]]]:
    import shlex # Import shlex at the top of the function
    log_suffix = f" in '{working_directory}'" if working_directory else ""
    logging.info(f"Attempting to execute shell command: {command}{log_suffix}")

    effective_cwd = None
    if working_directory:
        try:
            expanded_wd = os.path.expandvars(working_directory)
            abs_wd = os.path.abspath(expanded_wd)
            if os.path.isdir(abs_wd):
                effective_cwd = abs_wd
                logging.info(f"Effective working directory set to: {effective_cwd}")
            else:
                logging.error(f"Specified working directory does not exist or is not a directory: {abs_wd}")
                return -1, f"Invalid working directory: {working_directory} (Resolved to: {abs_wd})"
        except Exception as wd_err:
            logging.error(f"Error processing working directory '{working_directory}': {wd_err}")
            return -1, f"Error with working directory: {wd_err}"

    risky_keywords = [
        "rm ", "del ", "format ", "shutdown ", "reboot ", "sudo ",
        "remove-item ", "mkfs", ":(){:|:&};:",
        "npm install", "pip install", "conda install", # Added common package managers
        "npx create-react-app", "git clone", # Added common project initializers
    ]
    command_lower = command.lower().strip()
    is_risky = any(keyword in command_lower for keyword in risky_keywords)

    if is_risky and CONFIRM_RISKY_COMMANDS and allow_confirmation_prompt:
        logging.warning(f"[WARNING] Command '{command}' identified as potentially risky.")
        confirmation_question = (
            f"[RISKY COMMAND]\nThe agent wants to run:\n`{command}`"
            f"{f' in directory: {effective_cwd}' if effective_cwd else ''}\n\n"
            "Allow execution? (Type 'yes' to allow, anything else to deny)"
        )
        # Return code -2 signifies a need for user confirmation
        return -2, "Confirmation required for risky command.", {"type": "ask_user", "question": confirmation_question}
    elif is_risky and not allow_confirmation_prompt: # Risky command in a non-interactive context (e.g., multi_action)
        logging.warning(f"Risky command '{command}' encountered within a non-interactive context. Skipping confirmation and failing.")
        return -1, f"Risky command '{command}' cannot be confirmed in this context. Action failed."

    # --- Determine command type and prepare Python command if necessary ---
    is_python_command_execution = False
    final_command_for_python = command # Default to original command

    if command_lower.startswith(("python ", "python.exe ", "py ")):
        is_python_command_execution = True
        final_command_for_python = command # Already prefixed
    else:
        try: # shlex is now imported at the function top
            command_parts_for_py_check = shlex.split(command)
            if command_parts_for_py_check and command_parts_for_py_check[0].lower().endswith(".py"):
                # Further check if the .py file actually exists could be added here for more robustness
                # For example: if os.path.isfile(os.path.expandvars(command_parts_for_py_check[0])):
                is_python_command_execution = True
                python_exe_path = f'"{sys.executable}"' if " " in sys.executable else sys.executable
                # Reconstruct the command string with the Python executable prepended
                # The original 'command' string already has arguments and any necessary quoting from the LLM/user
                final_command_for_python = f"{python_exe_path} {command}"
                logging.info(f"Identified '{command_parts_for_py_check[0]}' as Python script. Prepared command: {final_command_for_python}")
        except Exception as e_shlex_prepare:
            logging.warning(f"Error using shlex to parse command for .py check: {e_shlex_prepare}. Original command: {command}")

    background_check_commands = [
        "npx", "npm", "yarn", "pip", "conda", "git", "docker",
        "make", "mvn", "gradle", "dotnet build", # Common build tools
    ]
    gui_launchers = [ # Common GUI application launchers
        "code.exe", "code", "notepad.exe", "notepad", "cmd.exe", "cmd",
        "powershell.exe", "powershell", "explorer.exe", "explorer",
        "start " # "start" is a Windows shell command to launch apps
    ]
    run_gui_launcher = False
    run_in_background_check = False
    command_start_lower = command_lower.split()[0] if command_lower else ""
    if not is_python_command_execution: # Only check these if not already identified as a Python script execution
        for gui_cmd in gui_launchers:
            # Check if command starts with a known GUI launcher name or if it's 'start' followed by arguments
            # Add cmd and powershell to the list of GUI launchers
            # so they are not treated as blocking commands
            if (command_lower.startswith(gui_cmd) or \
                os.path.basename(command_start_lower) == gui_cmd.strip() or \
                (gui_cmd == "start " and command_lower.startswith("start ") and len(command.split()) > 1)):
                run_gui_launcher = True
                logging.info(f"Identified command '{command}' as GUI launcher. Will use Popen with DEVNULL.")
                break
        if not run_gui_launcher: # If not a GUI launcher, check if it's a background-able command
            if any(command_start_lower == bg_cmd for bg_cmd in background_check_commands):
                run_in_background_check = True
                logging.info(f"Identified command '{command}' for background check. Will use Popen with PIPE.")
    # --- Command Execution Logic ---
    process = None
    try:
        if is_python_command_execution:
            if sys.platform == "win32":
                logging.info(f"Using Popen (CREATE_NEW_CONSOLE) for Python command on Windows: {final_command_for_python}{log_suffix}")
                try:
                    process = subprocess.Popen(final_command_for_python, shell=True,
                                               cwd=effective_cwd,
                                               env=os.environ.copy(),
                                               creationflags=subprocess.CREATE_NEW_CONSOLE)
                    time.sleep(0.5) # Give the new console a moment
                    logging.info(f"Launched Python script in new console window (PID: {process.pid}). User interaction may be required there.")
                    process = None # Detach from the process
                    return 0, f"Launched Python script '{command}' in a new visible terminal window."
                except Exception as e:
                    logging.error(f"Error launching Python script '{final_command_for_python}' in new console: {e}", exc_info=True)
                    return -1, f"Error launching Python script '{command}' in new console: {e}"
            else: # Non-Windows Python script execution
                # For non-Windows, if a Python script is meant to be a long-running server,
                # this Popen().wait() will block. This is standard for scripts expected to terminate.
                # If non-blocking is needed for Python scripts on Linux/macOS, further logic changes would be required.
                logging.info(f"Using Popen (blocking wait) for Python command on {sys.platform}: {final_command_for_python}{log_suffix}")
                try: # shlex is now imported at the function top
                    cmd_list_for_popen = shlex.split(final_command_for_python)
                    process = subprocess.Popen(cmd_list_for_popen, shell=False, # shell=False is safer with list
                                               cwd=effective_cwd,
                                               env=os.environ.copy())
                    exit_code = process.wait() # Wait for the script to complete
                    logging.info(f"Interactive Python command '{final_command_for_python}' finished with exit code: {exit_code}")
                    output_message = f"Python script '{command}' finished with exit code {exit_code}."
                    if exit_code != 0:
                        output_message += " Check agent terminal for errors."
                    return exit_code, output_message
                except KeyboardInterrupt: # Handle Ctrl+C if the agent is run interactively
                    logging.warning(f"User interrupted interactive Python command: {final_command_for_python}")
                    if process and process.poll() is None: process.terminate()
                    return -1, "User interrupted Python script execution."
                except Exception as wait_err:
                    logging.error(f"Error running/waiting for interactive Python command '{final_command_for_python}': {wait_err}")
                    if process and process.poll() is None: process.kill()
                    return -1, f"Error during Python script execution: {wait_err}"
        elif run_gui_launcher:
            logging.info(f"Using Popen (DEVNULL) to launch GUI: {command}{log_suffix}")
            process = subprocess.Popen(command, shell=True, # shell=True is common for 'start' or direct exe paths
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       stdin=subprocess.DEVNULL, # Prevent hanging on input
                                       cwd=effective_cwd)
            time.sleep(1.0) # Give GUI app time to launch
            poll_result = process.poll() # Check if it exited immediately
            if poll_result is not None and poll_result != 0:
                logging.warning(f"GUI command '{command}' exited immediately with code {poll_result}. It might have failed to launch properly.")
            # For GUI launchers, we often don't wait for them to close.
            return 0, f"Launched GUI command: {command}" # Assume success if Popen doesn't immediately error
        elif run_in_background_check:
            logging.info(f"Using Popen (PIPE) for background check: {command}{log_suffix}")
            process = subprocess.Popen(command, shell=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       stdin=subprocess.DEVNULL,
                                       text=True, encoding='utf-8', errors='replace',
                                       env=os.environ.copy(), # Pass environment
                                       cwd=effective_cwd)
            short_timeout_for_quick_exit = 5.0 # Seconds to wait for quick commands
            try:
                # Try to get output if it finishes quickly
                stdout_data, stderr_data = process.communicate(timeout=short_timeout_for_quick_exit)
                exit_code = process.returncode
                stdout_str = stdout_data.strip() if stdout_data else ""
                stderr_str = stderr_data.strip() if stderr_data else ""
                output = (stdout_str + ("\n" + stderr_str if stdout_str and stderr_str else stderr_str)).strip()
                logging.info(f"Command '{command}' completed quickly with exit code: {exit_code}")
                if exit_code == 0:
                    return 0, f"Command OK (completed quickly). Output:\n{output}" if output else "Command OK (completed quickly). (No output)"
                else:
                    logging.error(f"Command failed quickly (Code: {exit_code}). Output:\n{output}" if output else f"Command failed quickly (Code: {exit_code}). (No output)")
                    return exit_code, f"Command failed quickly (Code: {exit_code}). Output:\n{output}" if output else f"Command failed quickly (Code: {exit_code}). (No output)"
            except subprocess.TimeoutExpired:
                # Command is still running, assume it's a long-running background task
                logging.info(f"Command '{command}' did not exit within {short_timeout_for_quick_exit}s. Assuming long-running background task (PID: {process.pid}).")
                process_pid = process.pid
                process = None # Detach from the process, let it run
                return 0, f"Started background command (still running, PID: {process_pid}): {command}"

        else: # General blocking commands
            logging.info(f"Using subprocess.run (blocking) for command: {command}{log_suffix}")
            blocking_timeout = 180 # Timeout for blocking commands
            # Determine if shell=True is needed (e.g., for pipes, wildcards on Windows)
            use_shell_for_run = sys.platform == "win32" or any(c in command for c in ["|", ">", "<", "*", "?", ";", "&&", "||"])
            command_to_run_list_or_str = command
            if not use_shell_for_run:
                try: # shlex is now imported at the function top
                    command_to_run_list_or_str = shlex.split(command)
                    logging.debug(f"Executing command with shell=False: {command_to_run_list_or_str}")
                except ValueError as shlex_err_run: # If shlex fails to parse
                    logging.warning(f"shlex failed to parse command ('{shlex_err_run}'), falling back to shell=True for subprocess.run.")
                    use_shell_for_run = True
                    command_to_run_list_or_str = command

            result = subprocess.run(
                command_to_run_list_or_str,
                shell=use_shell_for_run,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                check=False, # We check returncode manually
                timeout=blocking_timeout,
                env=os.environ.copy(), # Pass environment
                cwd=effective_cwd
            )
            stdout_str = result.stdout.strip() if result.stdout else ""
            stderr_str = result.stderr.strip() if result.stderr else ""

            if stderr_str: # Prioritize stderr for the main part of the message if it exists
                output = f"Error Output:\n{stderr_str}" + (f"\nStandard Output:\n{stdout_str}" if stdout_str else "")
            else:
                output = f"Standard Output:\n{stdout_str}" if stdout_str else "(No output)"

            logging.info(f"Blocking command exit code: {result.returncode}")
            if output and output != "(No output)": logging.debug(f"Blocking command output:\n{output}")
            else: logging.debug("Blocking command produced no output.")

            if result.returncode == 0:
                return 0, f"Command OK. {output}"
            else:
                logging.error(f"Blocking command '{command}' failed (Code: {result.returncode}). {output}")
                return result.returncode, f"Command '{command}' failed (Code: {result.returncode}). {output}"

    except FileNotFoundError:
        cmd_name_fnf = command.split()[0]
        logging.error(f"Command not found or not executable: '{cmd_name_fnf}' in command '{command}'{log_suffix}")
        return -1, f"Command not found/executable: {cmd_name_fnf}. Ensure it's in PATH or use full path."
    except PermissionError:
        logging.error(f"Permission denied executing: {command}{log_suffix}")
        return -1, f"Permission denied executing command: {command}"
    except subprocess.TimeoutExpired:
        # This case should ideally be caught by specific timeouts above, but as a fallback for .run
        logging.error(f"Blocking command timed out: {command}{log_suffix}")
        return -1, f"Command timed out."
    except Exception as e:
        logging.error(f"Error executing shell command '{command}'{log_suffix}: {e}", exc_info=True)
        return -1, f"Error executing command: {e}"
    finally:
        # Ensure Popen processes that were not detached are cleaned up if they are still running
        if process and process.poll() is None: # If process exists and is still running
            try:
                logging.warning(f"Attempting to terminate lingering Popen process (PID: {process.pid}) for command '{command}'")
                process.terminate()
                try:
                    process.wait(timeout=2) # Give it a moment to terminate
                except subprocess.TimeoutExpired:
                    logging.warning(f"Process {process.pid} did not terminate gracefully, killing.")
                    process.kill() # Force kill if terminate fails
                # Close stdio pipes if they were opened (for Popen with PIPE)
                if hasattr(process, 'stdout') and process.stdout: process.stdout.close()
                if hasattr(process, 'stderr') and process.stderr: process.stderr.close()
            except Exception as e_cleanup:
                logging.warning(f"Error during Popen cleanup for '{command}': {e_cleanup}")






def execute_action(action: dict, agent: 'UIAgent') -> Union[Tuple[bool, str], Tuple[bool, str, Dict[str, Any]]]:
    """Execute an action and return the result."""
    action_type = action.get("action_type")
    parameters = action.get("parameters", {})
    
    logging.info(f"Executing action: {action_type}")
    logging.debug(f"Action parameters: {parameters}")
    
    try:
        if action_type == "process_local_files":
            file_path = parameters.get("file_path")
            prompt_param = parameters.get("prompt", "Analyze this file.")
            
            if not file_path:
                return False, "No file path provided", {"type": "error", "token_usage": ZERO_TOKEN_USAGE}
                
            success, result_or_error, token_usage = process_local_files([file_path], prompt_param)
            if not success:
                # result_or_error is an error string here
                return False, result_or_error, {"type": "error", "token_usage": token_usage} # token_usage from _process_local_files_action
            
            # result_or_error is a result_dict like {"status": "success", "result": "text...", ...}
            extracted_text_result = result_or_error.get("result", "")
            if not extracted_text_result and result_or_error.get("status") == "success":
                 extracted_text_result = "File processed successfully, but no textual result was extracted."

            return True, extracted_text_result, {"type": "file_processing_result", "token_usage": token_usage}
                
        elif action_type == "process_files_from_urls":
            url = parameters.get("url")
            prompt_param = parameters.get("prompt", "Analyze this file.") # Default prompt
            
            if not url:
                return False, "No URL provided", ZERO_TOKEN_USAGE
            
            # Convert single URL to list if needed
            urls = [url] if isinstance(url, str) else url if isinstance(url, list) else None
            if not urls:
                return False, "Invalid URL format provided", ZERO_TOKEN_USAGE
                
            # Call process_files_from_urls and return its raw result
            success, result_or_error, token_usage = process_files_from_urls(urls, prompt_param)
            if not success:
                # result_or_error is an error string, token_usage is from test_files.py (currently zero)
                return False, result_or_error, {"type": "error", "token_usage": token_usage}
            
            # Return just the raw result text, like search results
            # result_or_error is a dict, token_usage is from test_files.py (currently zero)
            return True, result_or_error.get("result", ""), {"type": "file_processing_result", "token_usage": token_usage}
                
        elif action_type == "edit_image_with_file":
            prompt = parameters.get("prompt")
            image_path = parameters.get("image_path")
            output_path = parameters.get("output_path", "output.png")
            
            result = generate_or_edit_image(prompt, image_path, output_path)
            
            if result.get("success"):
                # Return both success status and the full result dictionary
                return True, result["message"], {
                    "type": "image_generation_result",
                    "details": {
                        "success": True,
                        "image_data": result.get("image_data"),
                        "output_path": result.get("output_path")
                    }
                }
            else:
                return False, result.get("message", "Image generation failed"), {
                    "type": "image_generation_result",
                    "details": {
                        "success": False,
                        "error": result.get("error", "Unknown error")
                    }
                }

        logging.info(f"Executing action: {action_type} with params: {parameters}")

        # Default token usage for actions that don't make LLM calls or if call fails early
        action_token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

        try:
            if action_type == "multi_action":
                sequence = parameters.get("sequence")
                if not isinstance(sequence, list) or not sequence:
                    return False, "Action 'multi_action' failed: Invalid or empty 'sequence' parameter.", None
                all_messages = []
                overall_success = True
                # Tokens for multi_action itself are not from an LLM call here,
                # but sub-actions might have token usage passed up.
                # We'll let sub-actions return their tokens via special_directive.
                for i, sub_action_item in enumerate(sequence):
                    sub_action_type = sub_action_item.get("action_type", "N/A")
                    logging.info(f"[Multi-Action {i+1}/{len(sequence)}] Executing: {sub_action_type}")

                    current_sub_action = sub_action_item.copy()
                    current_sub_action["_is_top_level_action_"] = False

                    sub_result = execute_action(current_sub_action, agent)

                    sub_directive_tokens = {}
                    if len(sub_result) == 3 and isinstance(sub_result[2], dict):
                        sub_directive_tokens = sub_result[2].get("token_usage", {}) # type: ignore
                        # We don't accumulate here, iterative_task_executor will if this is top-level.
                        # If multi_action is nested, this detail might be lost unless explicitly handled.

                    if len(sub_result) == 3 and isinstance(sub_result[2], dict) and sub_result[2].get("type") == "ask_user": # type: ignore
                        logging.error(f"[Multi-Action {i+1}/{len(sequence)}] 'ask_user' (or risky command confirmation) inside 'multi_action' is not supported. Stopping sequence.")
                        all_messages.append(f"Step {i+1} ({sub_action_type}): FAIL - 'ask_user' or confirmation prompt cannot be handled within 'multi_action'.")
                        overall_success = False
                        break
                    elif len(sub_result) == 2:
                        sub_success, sub_message = sub_result # type: ignore
                    elif len(sub_result) == 3:
                        sub_success, sub_message, _ = sub_result # type: ignore
                    else:
                        sub_success, sub_message = False, "Unexpected sub-action result format"
                    all_messages.append(f"Step {i+1} ({sub_action_type}): {'OK' if sub_success else 'FAIL'} - {sub_message}")
                    if not sub_success:
                        logging.error(f"[Multi-Action {i+1}/{len(sequence)}] Failed: {sub_message}. Stopping sequence.")
                        overall_success = False
                        break
                    time.sleep(0.2)
                final_message_multi = "Multi-Action Sequence Result:\n" + "\n".join(all_messages)
                # For multi_action, the special_directive would be complex to aggregate.
                # We assume iterative_task_executor handles tokens from the planner's decision to use multi_action.
                return overall_success, final_message_multi, None

            elif action_type == "focus_window":
                title_sub = parameters.get("title_substring")
                if not title_sub: return False, "Action 'focus_window' failed: Missing 'title_substring'.", None

                success = focus_window_by_title(title_sub)
                message = f"Focus attempt on window containing '{title_sub}' {'succeeded' if success else 'failed'}."
                return success, message, None

            elif action_type == "click":
                desc = parameters.get("element_description")
                if not desc: return False, "Action 'click' failed: Missing 'element_description'.", None
                # locate_and_click_ui_element calls agent.select_ui_element_for_click which returns tokens.
                # These tokens need to be passed up.
                click_success, click_message_and_tokens = locate_and_click_ui_element(desc, agent) # type: ignore
                click_message = click_message_and_tokens
                click_tokens = action_token_usage # Default
                if isinstance(click_message_and_tokens, tuple) and len(click_message_and_tokens) == 2:
                    click_message, click_tokens = click_message_and_tokens # type: ignore

                return click_success, click_message, {"type": "click_result", "token_usage": click_tokens}


            elif action_type == "type":
                text = parameters.get("text_to_type")
                if text is None:
                    return False, "Action 'type' (as paste) failed: Missing 'text_to_type'.", None
                log_text_display = (text[:50] + '...') if len(text) > 53 else text
                try:
                    pyperclip.copy(text)
                    pyautogui.hotkey('ctrl', 'v')
                    logging.info(f"Pasted text: '{log_text_display}'")
                    return True, f"Pasted text: '{log_text_display}'", None
                except ImportError:
                    logging.warning("Module 'pyperclip' not found. Falling back to direct (fast) typing for 'type' action.")
                    try:
                        pyautogui.write(text)
                        logging.info(f"Typed text (fast, pyperclip not found): '{log_text_display}'")
                        return True, f"Typed text (fast, pyperclip not found): '{log_text_display}'", None
                    except Exception as e_write:
                        logging.error(f"Error during fallback pyautogui.write: {e_write}", exc_info=True)
                        return False, f"Action 'type' (fallback typing) failed: {e_write}", None
                except Exception as e_paste:
                    logging.error(f"Error during paste operation (pyperclip or hotkey): {e_paste}", exc_info=True)
                    logging.warning("Paste operation failed. Falling back to direct (fast) typing.")
                    try:
                        pyautogui.write(text)
                        logging.info(f"Typed text (fast, paste failed): '{log_text_display}'")
                        return True, f"Typed text (fast, paste failed): '{log_text_display}'", None
                    except Exception as e_write_fallback:
                        logging.error(f"Error during fallback pyautogui.write after paste error: {e_write_fallback}", exc_info=True)
                        return False, f"Action 'type' (paste and fallback typing) failed: {e_write_fallback}", None

            elif action_type == "press_keys":
                keys_input = parameters.get("keys")
                if not keys_input: return False, "Action 'press_keys' failed: Missing 'keys'.", None
                processed_keys = []
                original_keys_for_log = []
                try:
                    if isinstance(keys_input, str):
                        keys_to_process = [keys_input]
                        original_keys_for_log = [keys_input] # type: ignore
                    elif isinstance(keys_input, list):
                        keys_to_process = keys_input
                        original_keys_for_log = keys_input # type: ignore
                    else:
                        return False, f"Action 'press_keys' failed: 'keys' must be str or list, got {type(keys_input)}.", None
                    for key_item in keys_to_process:
                        if isinstance(key_item, str):
                            split_keys = [k.strip().lower() for k in key_item.split('+') if k.strip()]
                            processed_keys.extend(split_keys)
                        else:
                            logging.warning(f"Ignoring non-string item in 'keys' list: {key_item}")
                            continue
                    if not processed_keys:
                        return False, f"Action 'press_keys' failed: No processable keys found in input: {original_keys_for_log}", None
                    valid_keys = []
                    if 'pyautogui' in sys.modules:
                        valid_keys = [k for k in processed_keys if k in pyautogui.KEYBOARD_KEYS] # type: ignore
                        if len(valid_keys) != len(processed_keys):
                            invalid_found = [orig for orig in processed_keys if orig not in valid_keys]
                            logging.warning(f"Invalid or unknown keys found: {invalid_found}. Using valid: {valid_keys}")
                    else:
                        logging.error("PyAutoGUI not imported, cannot validate keys for 'press_keys'.")
                        return False, "Action 'press_keys' failed: PyAutoGUI module not available.", None
                    if not valid_keys:
                        return False, f"Action 'press_keys' failed: No valid pyautogui keys found after processing input: {original_keys_for_log}", None
                    if len(valid_keys) == 1:
                        pyautogui.press(valid_keys[0])
                        return True, f"Pressed key: {valid_keys[0]}", None
                    else:
                        pyautogui.hotkey(*valid_keys)
                        return True, f"Pressed keys combination: {valid_keys}", None
                except Exception as e:
                    logging.error(f"Error during pyautogui.hotkey/press. Error: {e}", exc_info=True)
                    return False, f"Action 'press_keys' failed: {e}", None

            elif action_type == "move_mouse":
                x = parameters.get("x")
                y = parameters.get("y")
                duration = parameters.get("duration_seconds", 0.25)
                if x is None or y is None: return False, "Action 'move_mouse' failed: Missing 'x' or 'y'.", None
                try:
                    x_coord, y_coord, duration_f = int(x), int(y), float(duration) # type: ignore
                    if duration_f < 0: duration_f = 0.0
                    pyautogui.moveTo(x_coord, y_coord, duration=duration_f)
                    return True, f"Mouse moved to ({x_coord}, {y_coord}).", None
                except (ValueError, TypeError) as e: return False, f"Action 'move_mouse' failed: Invalid params ({e}).", None
                except Exception as e: return False, f"Action 'move_mouse' failed: {e}", None

            elif action_type == "run_shell_command":
                command = parameters.get("command")
                working_directory = parameters.get("working_directory")
                if not command or not isinstance(command, str):
                    return False, "Action 'run_shell_command' failed: Missing/invalid 'command'.", None
                allow_confirm = parameters.get("_is_top_level_action_", True)
                shell_result = execute_shell_command(command, working_directory=working_directory, allow_confirmation_prompt=allow_confirm)
                if isinstance(shell_result, tuple) and len(shell_result) == 3 and shell_result[0] == -2: # type: ignore
                    return shell_result # type: ignore
                exit_code, output_msg = shell_result # type: ignore
                return (exit_code == 0), output_msg, None

            elif action_type == "run_python_script":
                script_path = parameters.get("script_path")
                working_directory = parameters.get("working_directory")
                if not script_path or not isinstance(script_path, str):
                    return False, "Action 'run_python_script' failed: Missing or invalid 'script_path'.", None
                python_executable = "python" if sys.platform != "win32" else "py"
                command_to_run = f'{python_executable} "{script_path}"'
                logging.info(f"Constructed Python command: {command_to_run}")

                py_exec_result = execute_shell_command(command_to_run, working_directory=working_directory)
                if isinstance(py_exec_result, tuple) and len(py_exec_result) == 3 and py_exec_result[0] == -2: # type: ignore
                    return py_exec_result # type: ignore
                exit_code, output_msg = py_exec_result # type: ignore
                return (exit_code == 0), output_msg, None

            elif action_type == "navigate_web":
                url = parameters.get("url")
                if not url or not isinstance(url, str): return False, "Action 'navigate_web' failed: Missing/invalid 'url'.", None
                success, message = navigate_web(url)
                return success, message, None

            elif action_type == "search_web":
                query = parameters.get("query")
                if not query or not isinstance(query, str):
                    return False, "Action 'search_web' failed: Missing or invalid 'query' parameter.", {"type": "error", "token_usage": action_token_usage}
                success, search_api_message, token_usage_from_search = search_web_for_info(query)
                directive = {"type": "search_result", "summary": search_api_message, "token_usage": token_usage_from_search}
                if not success: directive["error"] = True
                return success, f"Web Search {'Result' if success else 'Failed'}: {search_api_message}", directive

            elif action_type == "search_youtube":
                query = parameters.get("query")
                if not query or not isinstance(query, str):
                    return False, "Action 'search_youtube' failed: Missing or invalid 'query' parameter.", None # No LLM call here directly
                logging.info(f"Initiating YouTube search and transcript analysis for: '{query}'")
                try:
                    store_success, store_message, videos_processed = process_and_store_youtube_videos(query, max_results=5)
                    if not store_success or not videos_processed:
                        logging.warning(f"Failed to process or find videos for query '{query}': {store_message}")
                        return False, store_message, None
                    logging.info(f"Successfully processed and stored {len(videos_processed)} videos for query '{query}'.")
                    logging.info(f"Performing transcript search in Qdrant for: '{query}'")
                    semantic_results = search_youtube_transcripts(query, n_results_per_video=2, fetch_limit=50)
                    if not semantic_results:
                        simple_result_text = f"I found and processed {len(videos_processed)} video(s) for '{query}', but no specific relevant segments in transcripts after analysis. Videos found:\n"
                        for v_info in videos_processed: simple_result_text += f"- {v_info['title']} ({v_info['url']})\n" # type: ignore
                        return True, simple_result_text, {"type": "youtube_search_result", "suggested_videos": videos_processed, "relevant_transcript_chunks": {}}
                    num_relevant_videos = len(semantic_results)
                    exec_msg = f"Found {num_relevant_videos} YouTube video(s) with relevant transcript segments for '{query}' after processing."
                    final_youtube_results = {"type": "youtube_search_result", "suggested_videos": videos_processed, "relevant_transcript_chunks": semantic_results}
                    return True, exec_msg, final_youtube_results
                except Exception as e:
                    logging.error(f"Error during 'search_youtube' action for query '{query}': {e}", exc_info=True)
                    return False, f"Error processing YouTube search: {str(e)}", None

            elif action_type == "wait":
                duration = parameters.get("duration_seconds")
                if duration is None: return False, "Action 'wait' failed: Missing 'duration_seconds'.", None
                try:
                    wait_time = float(duration) # type: ignore
                    if wait_time < 0: wait_time = 0
                    logging.info(f"Waiting for {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    return True, f"Waited for {wait_time:.2f} seconds.", None
                except ValueError: return False, f"Action 'wait' failed: Invalid 'duration_seconds': {duration}.", None

            elif action_type == "ask_user":
                question = parameters.get("question")
                if not question or not isinstance(question, str):
                    return False, "Action 'ask_user' failed: Missing/invalid 'question'.", None
                try:
                    logging.info(f"Action 'ask_user' triggered with question: '{question}'")
                    ask_info = {"type": "ask_user", "question": question}
                    return True, f"Waiting for user input for question: '{question}'", ask_info
                except Exception as e:
                    logging.warning(f"Error preparing 'ask_user': {e}")
                    return False, f"Action 'ask_user' failed during preparation: {e}", None

            elif action_type == "click_and_type":
                desc = parameters.get("element_description")
                text = parameters.get("text_to_type")
                interval = parameters.get("interval_seconds", 0.05)
                if not desc: return False, "Action 'click_and_type' failed: Missing 'element_description'.", None
                if text is None: return False, "Action 'click_and_type' failed: Missing 'text_to_type'.", None
                logging.info(f"[click_and_type] Performing click on: '{desc}'")

                click_success, click_message_and_tokens = locate_and_click_ui_element(desc, agent) # type: ignore
                click_message = click_message_and_tokens
                click_tokens = action_token_usage # Default
                if isinstance(click_message_and_tokens, tuple) and len(click_message_and_tokens) == 2:
                    click_message, click_tokens = click_message_and_tokens # type: ignore

                if not click_success:
                    logging.error(f"[click_and_type] Click part failed: {click_message}")
                    return False, f"Action 'click_and_type' failed during click: {click_message}", {"type": "click_failed", "token_usage": click_tokens}
                logging.info(f"[click_and_type] Click successful, now typing.")
                try:
                    interval_f = float(interval) # type: ignore
                    if interval_f < 0: interval_f = 0.0
                    time.sleep(0.3)
                    pyautogui.typewrite(text, interval=interval_f) # type: ignore
                    log_text = (text[:50] + '...') if len(text) > 53 else text # type: ignore
                    return True, f"Clicked '{desc}' and typed text: '{log_text}'", {"type": "click_and_type_success", "token_usage": click_tokens}
                except ValueError:
                    return False, f"Action 'click_and_type' failed during type: Invalid 'interval_seconds': {interval}.", {"type": "type_failed", "token_usage": click_tokens}
                except Exception as e:
                    return False, f"Action 'click_and_type' failed during type: {e}", {"type": "type_failed", "token_usage": click_tokens}

            elif action_type == "describe_screen":
                logging.info("Capturing screen to describe...")
                screenshot_pil = capture_full_screen()
                if screenshot_pil is None:
                    return False, "Action 'describe_screen' failed: Could not capture screen.", {"type": "error", "token_usage": action_token_usage}
                success, message, desc_tokens = get_screen_description_from_gemini(screenshot_pil, agent.model)
                directive = {"type": "screen_description", "description": message, "token_usage": desc_tokens} if success else {"type": "error", "token_usage": desc_tokens}
                return success, message, directive

            elif action_type == "write_file":
                file_path = parameters.get("file_path")
                content_to_write = parameters.get("content") # Renamed to avoid conflict
                append = parameters.get("append", False)
                if file_path is None or content_to_write is None:
                    logging.error(f"Missing 'file_path' or 'content' parameter for write_file action.")
                    return False, "Missing 'file_path' or 'content' parameter for write_file action.", None
                try:
                    expanded_path = os.path.expandvars(file_path) # type: ignore
                except Exception as path_err:
                    logging.error(f"Error expanding file path '{file_path}': {path_err}")
                    return False, f"Invalid file path: {file_path}", None
                success, message = _execute_write_file(expanded_path, content_to_write, append) # type: ignore
                return success, message, None

            elif action_type == "read_file":
                file_path = parameters.get("file_path")
                if not file_path or not isinstance(file_path, str):
                    logging.error("Missing or invalid 'file_path' parameter for read_file action.")
                    return False, "Missing or invalid 'file_path' for read_file.", None
                success, content_or_error = _execute_read_file(file_path)
                return success, content_or_error, None

            elif action_type == "start_visual_listener":
                # _execute_visual_listener returns tokens in its directive
                return _execute_visual_listener(parameters, agent, agent.model) # type: ignore

            elif action_type == "generate_large_content_with_gemini":
                context_summary = parameters.get("context_summary")
                detailed_prompt = parameters.get("detailed_prompt_for_gemini")
                target_file_path = parameters.get("target_file_path")

                if not all([context_summary, detailed_prompt, target_file_path]):
                    return False, "Action 'generate_large_content_with_gemini' failed: Missing one or more required parameters (context_summary, detailed_prompt_for_gemini, target_file_path).", {"type": "error", "token_usage": action_token_usage}

                logging.info(f"Calling external Gemini for large content generation. Target: {target_file_path}")
                gen_tokens = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
                try:
                    generation_model = model 
                    response = generation_model.generate_content(detailed_prompt) 
                    gen_tokens = _get_token_usage(response)
                    generated_content = ""
                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                         generated_content = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()

                    if not generated_content:
                        return False, "External Gemini generation returned empty content.", {"type": "generation_failed", "token_usage": gen_tokens}

                    write_success, write_message = _execute_write_file(target_file_path, generated_content, append=False) # type: ignore
                    return write_success, f"Generated content and {write_message}", {"type": "generation_complete", "file_path": target_file_path, "token_usage": gen_tokens}
                except Exception as e_gen:
                    logging.error(f"Error during 'generate_large_content_with_gemini': {e_gen}", exc_info=True)
                    return False, f"Error during large content generation: {e_gen}", {"type": "generation_error", "token_usage": gen_tokens}

            elif action_type == "INFORM_USER":
                msg = parameters.get("message", "")
                if not msg:
                    return False, "Action 'INFORM_USER' failed: Missing 'message'.", None
                return True, f"Message for user: {msg}", {"type": "inform_user", "message": msg}

            elif action_type == "capture_screenshot":
                file_path_param = parameters.get("file_path")
                logging.info("Capturing screenshot...")
                screenshot_pil = capture_full_screen()
                if screenshot_pil is None:
                    return False, "Action 'capture_screenshot' failed: Could not capture screen.", None
                save_path = None
                try:
                    if file_path_param and isinstance(file_path_param, str):
                        expanded_path = os.path.expandvars(file_path_param)
                        save_path = os.path.abspath(expanded_path)
                        directory = os.path.dirname(save_path)
                        if directory and not os.path.exists(directory):
                            os.makedirs(directory)
                            logging.info(f"Created directory for screenshot: {directory}")
                    else:
                        if 'DEBUG_DIR' not in globals() or not os.path.exists(DEBUG_DIR): os.makedirs(DEBUG_DIR, exist_ok=True) # type: ignore
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        save_path = os.path.join(DEBUG_DIR, f"screenshot_{timestamp}.png") # type: ignore
                    screenshot_pil.save(save_path, "PNG")
                    logging.info(f"Screenshot saved successfully to: {save_path}")
                    return True, f"Screenshot saved to {save_path}", None
                except OSError as e:
                    logging.error(f"Error saving screenshot to '{save_path}': {e}")
                    return False, f"Action 'capture_screenshot' failed: OS error saving file: {e}", None
                except Exception as e:
                    logging.error(f"Unexpected error saving screenshot to '{save_path}': {e}", exc_info=True)
                    return False, f"Action 'capture_screenshot' failed: Unexpected error saving file: {e}", None

            elif action_type == "task_complete":
                return True, "Task completion signaled.", None

            else:
                logging.warning(f"Unknown action type encountered: {action_type}")
                return False, f"Action failed: Unknown action type '{action_type}'.", None

        except Exception as e:
            logging.error(f"Unexpected error during execution of action '{action_type}': {e}", exc_info=True)
            return False, f"Action '{action_type}' failed due to runtime error: {e}", None

    except Exception as e:
        logging.error(f"Unexpected error during execution of action '{action_type}': {e}", exc_info=True)
        return False, f"Action '{action_type}' failed due to runtime error: {e}", None






def chat_with_user(
    conversation_history: List[str], 
    user_input: str, 
    execution_results: List[Tuple[dict, bool, str, Optional[Dict]]], 
    plan_reasoning: str
) -> Tuple[str, Dict[str, int]]:
    logging.info("Generating conversational response.")
    summary = "\nExecution Summary:\n"
    task_result_info = "" 

    if not execution_results:
        if plan_reasoning and "Plan generation failed" in plan_reasoning: summary += f"Plan generation failed.\nReason: {plan_reasoning}\n"
        elif plan_reasoning: summary += f"No actions executed (plan empty).\nPlanning Reasoning: {plan_reasoning}\n"
        else: summary += "No actions planned or executed.\n"
    else:
        total_steps = 0
        successful_steps = 0
        _ = "Task initiated." # final_outcome_message not used
        last_successful_result_action = None

        for action, success, message, _ in execution_results: 
            action_type = action.get('action_type', 'Unknown')

            if action_type != 'REPLAN':
                total_steps += 1
                if success:
                    successful_steps += 1
                    if action_type in ["search_web", "describe_screen", "run_shell_command", "run_python_script", "search_youtube", "generate_large_content_with_gemini"]: # Added generate_large_content_with_gemini
                        logging.debug(f"Found potential result from successful action '{action_type}': {message}")
                        last_successful_result_action = (action_type, message) 

        summary += f"Attempted {total_steps} step(s), {successful_steps} succeeded.\n"
        last_action, last_success, last_msg, _ = execution_results[-1] 
        status = "[OK] Completed" if last_success and last_action.get('action_type') != 'REPLAN' else "[ERROR] Failed"
        action_type = last_action.get('action_type', 'Unknown')
        short_message = (str(last_msg)[:200] + '...') if len(str(last_msg)) > 203 else str(last_msg) 
        summary += f"Final Step ({action_type}): {status} - {short_message}\n"

        if last_successful_result_action:
            result_action_type, result_message = last_successful_result_action
            logging.debug(f"Processing result from '{result_action_type}': {result_message}")
            if result_action_type == "search_youtube":
                if isinstance(result_message, dict):
                    youtube_data = result_message
                    formatted_yt_info_parts = ["YouTube Search & Transcript Analysis Results:"]
                    if youtube_data.get("suggested_videos"):
                        formatted_yt_info_parts.append("\nSuggested Videos (Top 3):")
                        for video in youtube_data["suggested_videos"][:3]: 
                            formatted_yt_info_parts.append(f"- [{video['title']}]({video['url']})") 
                    if youtube_data.get("relevant_transcript_chunks"):
                        formatted_yt_info_parts.append("\nRelevant Information from Transcripts (Top chunks):")
                        chunk_count = 0
                        for _, video_data in youtube_data["relevant_transcript_chunks"].items(): # video_id not used
                            if chunk_count >= 3: break 
                            video_title_yt = video_data.get("title", "Unknown Video")
                            _ = next((vid_info['url'] for vid_info in youtube_data.get("suggested_videos", []) if video_title_yt == vid_info['title']), "#") # video_url_yt not used
                            formatted_yt_info_parts.append(f"  From Video: {video_title_yt}") 
                            for chunk in video_data.get("chunks", [])[:1]: 
                                if chunk_count >= 3: break
                                formatted_yt_info_parts.append(f"    - Chunk: \"...{chunk.get('content', '')[:100]}...\" (Score: {chunk.get('score', 0):.2f})")
                                chunk_count += 1
                    task_result_info = "\n".join(formatted_yt_info_parts)
                else:
                    task_result_info = f"YouTube search returned data in unexpected format: {type(result_message)}"
            elif result_action_type == "search_web":
                if "Web Search Result: " in result_message:
                    task_result_info = result_message.split("Web Search Result: ", 1)[-1]
                else: 
                    task_result_info = result_message
            elif result_action_type == "describe_screen":
                if "Screen description: " in result_message:
                    task_result_info = result_message.split("Screen description: ", 1)[-1]
                else:
                    task_result_info = result_message
            elif result_action_type == "generate_large_content_with_gemini":
                if "Generated content and Successfully write to file:" in result_message:
                    file_path_gen = result_message.split("Generated content and Successfully write to file: ")[-1]
                    task_result_info = f"Content was generated and saved to {file_path_gen}."
                else:
                    task_result_info = result_message # Fallback to full message
            elif result_action_type in ["run_shell_command", "run_python_script"]:
                if "Command OK. Standard Output:" in result_message:
                    task_result_info = result_message.split("Command OK. Standard Output:", 1)[-1].strip()
                elif "Command OK. Error Output:" in result_message: 
                    task_result_info = result_message.split("Command OK. Error Output:", 1)[-1].strip()
                elif "Command OK." in result_message and "(No output)" not in result_message and "completed quickly" not in result_message: 
                    task_result_info = result_message.split("Command OK.", 1)[-1].strip()
            else: 
                task_result_info = result_message

            if task_result_info:
                task_result_info = task_result_info.strip() 
                logging.info(f"Formatted Task Result Info: '{task_result_info}'") 
            else:
                logging.warning(f"Result message from '{result_action_type}' was present but resulted in empty task_result_info after formatting.")
        else:
            logging.info("No successful result-producing action found in execution results.")

    max_history_turns = 10
    truncated_history = conversation_history[-(max_history_turns*2):]
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in truncated_history]) 

    logging.debug(f"Final task_result_info being passed to chat prompt: '{task_result_info}'")

    prompt = f"""
You are a helpful and concise Assistant. Your primary goal is to inform the user of the task outcome.

Conversation History:
{history_str}

User's Last Instruction: "{user_input}"

Execution Outcome Summary:
{summary}
---
Task Result Found:
{task_result_info if task_result_info else "None"}
---

**CRITICAL INSTRUCTION:** Generate a brief, friendly, informative response (1-3 sentences) based ONLY on the information above.

*   **IF 'Task Result Found' includes YouTube video information (especially 'Relevant Information from Transcripts'):**
    *   Incorporate the key findings from the transcript chunks into your answer if they are relevant to the user's query.
    *   When you use information from a YouTube video, cite it by providing the video title as a clickable link (the format `Video Title` is already provided in the 'Task Result Found'). For example: "According to the video 'Video Title', you should..." or "I found some useful tips in 'Video Title' regarding...".
    *   You can also suggest relevant videos directly if the user might find them helpful, e.g., "You might find this video helpful: Video Title".
*   **ELSE IF 'Task Result Found' is NOT 'None' (and not YouTube specific):** Your response **MUST** clearly state this result to the user. Start with something like "Okay, here's the score:" or "I found this:" or directly state the information (e.g., "The score was 3-2."). **DO NOT just say 'Task completed'.**
*   **IF 'Task Result Found' IS 'None' AND the task succeeded:** Confirm completion concisely (e.g., "Okay, I've completed that action.").
*   **IF the task failed:** Acknowledge failure, briefly state the issue from the summary. Suggest checking logs if complex.
*   **IF plan generation failed or the plan was empty:** State that based on the summary.

Focus ONLY on the outcome of the *last* user instruction. Do NOT invent info or ask unprompted questions.

Generate the response now:
"""
    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0} 
    try:
        safety_settings = { 'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE', 'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE', 'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE', 'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE' }
        response = model.generate_content(prompt, safety_settings=safety_settings) 
        token_usage = _get_token_usage(response)
        reply = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            reply = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()

        if not reply:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason
                logging.error(f"LLM chat response blocked: {reason}")
                reply = f"My response was blocked ({reason}). Check logs for execution details."
            else:
                logging.error("LLM returned empty chat response.")
                reply = "Processed request. Check logs for details."
        else:
            logging.info(f"Generated reply: {reply}")
        return reply, token_usage
    except ValueError as ve:
        logging.error(f"ValueError accessing LLM chat response: {ve}")
        block_reason_str = ""
        if "response.prompt_feedback" in str(ve) and hasattr(response, 'prompt_feedback'): 
            try: block_reason_str = f" (Block Reason: {response.prompt_feedback.block_reason})" # type: ignore
            except Exception: pass
        return f"Issue generating response{block_reason_str}. Check logs.", token_usage 
    except Exception as e:
        logging.error(f"Error generating chat response: {e}", exc_info=True) 
        if execution_results:
            _, last_success, _, _ = execution_results[-1] 
            if last_success:
                if task_result_info:
                    return task_result_info, token_usage 
                else:
                    return "Okay, completed the actions.", token_usage 
            else:
                return "Encountered issues. Check logs for details.", token_usage 
        else:
            return "Couldn't process request fully. Check logs or rephrase.", token_usage





def assess_action_outcome(
    original_instruction: str,
    action: dict,
    exec_success: bool,
    exec_message: str,
    screenshot_after: Optional[Image.Image],
    llm_model: genai.GenerativeModel,
    screenshot_before_hash: Optional[str] = None
) -> Tuple[str, str, Dict[str, int]]:
    action_type = action.get("action_type", "unknown")
    logging.info(f"Assessing outcome for action: {action_type}")
    logging.debug(f"Execution Result: Success={exec_success}, Message='{exec_message}'")

    token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0} 

    if not exec_success:
        logging.warning(f"Execution reported failure. Assessing as FAILURE. Reason: {exec_message}")
        return "FAILURE", f"Action execution failed: {exec_message}", token_usage

    non_visual_actions_or_explicit_success = {
        "wait", "get_clipboard", "set_clipboard", "save_credential", "read_file", "write_file",
        "navigate_web", "search_web", "capture_screenshot", "INFORM_USER", "search_youtube",
        "generate_large_content_with_gemini" # Added this
    }
    if action_type in non_visual_actions_or_explicit_success or \
    action_type in {"run_shell_command", "run_python_script"}:
        logging.info(f"Action '{action_type}' reported success and is non-visual or has explicit success criteria. Assessing as SUCCESS.")
        return "SUCCESS", f"Action executed successfully: {exec_message}", token_usage

    if screenshot_after is None:
        logging.warning("Cannot perform visual assessment: Screenshot after action is missing. Assuming success based on execution report.")
        return "SUCCESS", f"Action executed successfully (non-visual assessment as screenshot_after is missing): {exec_message}", token_usage

    current_screenshot_hash = _hash_pil_image(screenshot_after)
    screen_changed = False
    visual_change_expected = action_type in {"click", "type", "press_keys", "focus_window", "click_and_type", "move_mouse"}

    if screenshot_before_hash and current_screenshot_hash:
        if screenshot_before_hash != current_screenshot_hash:
            screen_changed = True
            logging.info("Screen content hash changed after the action.")
        else:
            if visual_change_expected:
                logging.warning(f"Screen content hash did NOT change after action '{action_type}' which usually causes visual changes. Assessing as FAILURE.")
                return "FAILURE", f"Action '{action_type}' reported success, but screen content did not change visually, indicating it likely failed.", token_usage
            else:
                logging.info("Screen content hash did not change (as expected for this action type or no change expected).")
    elif visual_change_expected:
        logging.info("No prior screenshot hash available to compare for visual change detection.")


    screenshot_data = image_to_base64(screenshot_after)
    if not screenshot_data:
        logging.error("Failed to convert screenshot_after to base64 for LLM assessment. Assuming success based on execution report.")
        return "SUCCESS", f"Action executed successfully (screenshot conversion failed for LLM assessment): {exec_message}", token_usage

    action_params = action.get("parameters", {})
    action_desc = f"Action Type: {action_type}, Parameters: {action_params}"
    action_goal = f"The goal was related to the action: {action_type}."
    if action_type == 'click':
        action_goal = f"The goal was to click on '{action_params.get('element_description', 'unspecified element')}'."
    elif action_type == 'type':
        action_goal = f"The goal was to type text starting with '{action_params.get('text_to_type', '')[:20]}...'."
    elif action_type == 'focus_window':
        action_goal = f"The goal was to bring the window with title containing '{action_params.get('title_substring', '')}' to the foreground."
    elif action_type == 'click_and_type':
        action_goal = f"The goal was to click on '{action_params.get('element_description', 'unspecified element')}' and then type text starting with '{action_params.get('text_to_type', '')[:20]}...'."

    prompt_parts = [
        "You are an expert QA agent assessing the outcome of an automated action.",
        f"The user's overall objective was: '{original_instruction}'",
        f"The action attempted was: {action_desc}",
        f"{action_goal}",
        f"The execution system reported: Success={exec_success}, Message='{exec_message}'",
        "The current screen *after* the action is provided as an image.",
        f"Did the screen change visually after the action? {'Yes' if screen_changed else 'No' if screenshot_before_hash and not screen_changed else 'Unknown (no comparison or no change expected)'}",
        "\n--- Task ---",
        "Based *only* on the provided information (action details, execution report, and the *after* screenshot), assess if the action *successfully achieved its intended sub-goal*.",
        "Consider:",
        "  - Does the screen *after* the action reflect the expected outcome? (e.g., If 'click File menu', is the File menu open? If 'type hello', is 'hello' visible?).",
        "  - Does the execution message align with the visual evidence?",
        "  - If the screen didn't change visually when it was expected to, the action likely failed, even if the execution system reported success.",
        "Provide step-by-step reasoning for your assessment.",
        "\n--- Output Format ---",
        "Provide ONLY these two lines:",
        "REASONING: [Your detailed step-by-step reasoning here]",
        "ASSESSMENT: [Choose ONE: SUCCESS, FAILURE, RETRY_POSSIBLE]"
    ]

    content = [
        {"inline_data": {"mime_type": "image/png", "data": screenshot_data}},
        {"text": "\n".join(prompt_parts)}
    ]

    try:
        safety_settings = {}
        response = llm_model.generate_content(content, safety_settings=safety_settings)
        token_usage = _get_token_usage(response)
        response_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()

        if not response_text:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                logging.error(f"LLM assessment response blocked: {block_reason}")
                return "FAILURE", f"Assessment failed: LLM response blocked ({block_reason}). Assuming failure.", token_usage
            else:
                logging.error("LLM assessment response was empty.")
                return "FAILURE", "Assessment failed: LLM returned empty response. Assuming failure.", token_usage

        logging.debug(f"LLM Raw Assessment Response:\n{response_text}")

        reasoning = "Assessment reasoning not extracted."
        assessment = "FAILURE"

        reasoning_match = re.search(r'REASONING:\s*(.*?)(?=\nASSESSMENT:|$)', response_text, re.DOTALL | re.IGNORECASE)
        assessment_match = re.search(r'ASSESSMENT:\s*(SUCCESS|FAILURE|RETRY_POSSIBLE)', response_text, re.IGNORECASE)

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            assessment_keyword_pos = response_text.upper().find("ASSESSMENT:")
            reasoning = response_text[:assessment_keyword_pos].strip() if assessment_keyword_pos != -1 else response_text

        if assessment_match:
            assessment = assessment_match.group(1).upper()
            if assessment not in ["SUCCESS", "FAILURE", "RETRY_POSSIBLE"]:
                logging.warning(f"Extracted assessment status '{assessment}' is invalid. Defaulting to FAILURE.")
                assessment = "FAILURE"
        else:
            logging.warning(f"Could not parse ASSESSMENT status from response. Defaulting to FAILURE.\nResponse: {response_text}")
            reasoning += "\n(Agent Note: Could not parse ASSESSMENT status, defaulted to FAILURE)"

        if assessment == "SUCCESS" and visual_change_expected and not screen_changed and screenshot_before_hash:
            logging.warning(f"Overriding LLM SUCCESS assessment to FAILURE because screen hash did not change for visual action '{action_type}'.")
            assessment = "FAILURE"
            reasoning += f"\n(Agent Override: Assessment changed to FAILURE because screen content did not change visually for action '{action_type}'.)"

        logging.info(f"LLM Assessment: {assessment}. Reasoning: {reasoning}")
        return assessment, reasoning, token_usage

    except Exception as e:
        logging.error(f"Error during assessment LLM call or parsing: {e}", exc_info=True)
        return "FAILURE", f"Assessment failed due to an unexpected error: {e}. Assuming failure.", token_usage
