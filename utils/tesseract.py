import pytesseract 
import logging 
import os
import sys
import shutil

PROJECT_TESSERACT_DIR_NAME = "Tesseract-OCR"
TESSERACT_EXE_NAME = "tesseract.exe"

def ensure_tesseract_windows():
    """
    Checks for Tesseract ONLY in the project's "Tesseract-OCR" folder,
    relative to this script's location.
    If found, configures pytesseract and PATH.
    If not found, logs an error and exits.
    Does NOT attempt any installation.
    """
    try:
        script_file_path = os.path.abspath(__file__)
        utils_dir = os.path.dirname(script_file_path)
        project_root_dir = os.path.dirname(utils_dir)
        project_tesseract_exe_path = os.path.join(project_root_dir, PROJECT_TESSERACT_DIR_NAME, TESSERACT_EXE_NAME)
        expected_tesseract_dir_abs = os.path.join(project_root_dir, PROJECT_TESSERACT_DIR_NAME)

        if os.path.exists(project_tesseract_exe_path):
            logging.info(f"Found Tesseract in project-specific folder: {project_tesseract_exe_path}")
            tesseract_exe_dir = os.path.dirname(project_tesseract_exe_path)
            if tesseract_exe_dir not in os.environ.get("PATH", "").split(os.pathsep):
                 os.environ["PATH"] = tesseract_exe_dir + os.pathsep + os.environ.get("PATH","")
                 logging.info(f"Temporarily added {tesseract_exe_dir} to PATH.")
            pytesseract.pytesseract.tesseract_cmd = project_tesseract_exe_path
            print(f"[OK] Tesseract found in project folder ({project_tesseract_exe_path}) and configured.")
            return True
        else:
            logging.error(f"Tesseract not found at the expected project path: {project_tesseract_exe_path}")
            print(f"\n[ERROR] FATAL ERROR: Tesseract OCR not found in the project directory.")
            print(f"Please ensure Tesseract is located at: {expected_tesseract_dir_abs}")
            print(f"Specifically, '{TESSERACT_EXE_NAME}' should be in that folder.")
            sys.exit(1)

    except NameError: 
        logging.error("Could not determine project-specific Tesseract path (__file__ not defined).")
        print("\n[ERROR] FATAL ERROR: Tesseract OCR could not be found or configured.")
        print("Unable to determine the script's location to find the Tesseract-OCR folder.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while trying to locate Tesseract: {e}", exc_info=True)
        print("\n[ERROR] FATAL ERROR: Tesseract OCR could not be found or configured due to an unexpected error.")
        sys.exit(1)
