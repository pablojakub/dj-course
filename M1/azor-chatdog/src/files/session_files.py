import os
import json
import re
from datetime import datetime
from typing import List, Any, Dict
from files.config import LOG_DIR

def generate_friendly_filename(first_prompt: str, session_id: str, max_length: int = 50) -> str:
    """
    Generates a friendly filename based on the first user prompt.

    Args:
        first_prompt: The first user message in the conversation
        session_id: The session UUID for uniqueness
        max_length: Maximum length for the prompt part of filename

    Returns:
        str: A sanitized filename in format: {sanitized_prompt}_{short_id}.json
    """
    # Take first N characters of the prompt
    prompt_part = first_prompt[:max_length].strip()

    # Remove or replace invalid filename characters
    # Keep only letters, numbers, spaces, and basic punctuation
    prompt_part = re.sub(r'[^\w\s\-]', '', prompt_part)

    # Replace spaces with underscores and collapse multiple underscores
    prompt_part = re.sub(r'\s+', '_', prompt_part)
    prompt_part = re.sub(r'_+', '_', prompt_part)

    # Remove leading/trailing underscores
    prompt_part = prompt_part.strip('_')

    # If prompt is empty after sanitization, use default
    if not prompt_part:
        prompt_part = "conversation"

    # Use first 8 characters of UUID for uniqueness
    short_id = session_id[:8]

    return f"{prompt_part}_{short_id}"

def get_session_display_name(session_id: str) -> str:
    """
    Gets a friendly display name for a session.

    Args:
        session_id: The full session UUID

    Returns:
        str: Friendly display name if available, otherwise short ID
    """
    session_file = find_session_file(session_id)
    if not session_file:
        return session_id[:8]

    filename = os.path.basename(session_file)

    # If it's old format, return short ID
    if filename.endswith('-log.json'):
        return session_id[:8]

    # If it's new format, extract friendly name
    base_name = filename[:-5]  # Remove .json
    if '_' in base_name:
        parts = base_name.split('_')
        friendly_name = '_'.join(parts[:-1])
        return friendly_name

    return session_id[:8]

def find_session_file(session_id: str) -> str | None:
    """
    Finds the session file by session_id. Handles both old format (UUID-log.json)
    and new format (friendly_name_shortid.json).

    Args:
        session_id: The full session UUID

    Returns:
        str | None: Full path to the session file, or None if not found
    """
    if not os.path.exists(LOG_DIR):
        return None

    # Try old format first (for backward compatibility)
    old_format_path = os.path.join(LOG_DIR, f"{session_id}-log.json")
    if os.path.exists(old_format_path):
        return old_format_path

    # Try new format - search for files ending with short_id
    short_id = session_id[:8]
    for filename in os.listdir(LOG_DIR):
        if filename.endswith(f"_{short_id}.json"):
            return os.path.join(LOG_DIR, filename)

    return None

def load_session_history(session_id: str) -> tuple[List[Dict], str | None]:
    """
    Loads session history from a JSON file in universal format.

    Returns:
        tuple[List[Dict], str | None]: (conversation_history, error_message)
        History format: [{"role": "user|model", "parts": [{"text": "..."}]}, ...]
    """

    log_filename = find_session_file(session_id)
    if not log_filename:
        return [], f"Session log file for session '{session_id[:8]}...' does not exist. Starting new session."

    try:
        with open(log_filename, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except json.JSONDecodeError:
        return [], f"Cannot decode log file '{log_filename}'. Starting new session."

    # Convert JSON data to universal format (dictionaries)
    # This format works with both Gemini and LLaMA clients
    history = []
    for entry in log_data.get('history', []):
        content = {
            "role": entry['role'],
            "parts": [{"text": entry['text']}]
        }
        history.append(content)

    return history, None

def save_session_history(session_id: str, history: List[Dict], system_prompt: str, model_name: str) -> tuple[bool, str | None]:
    """
    Saves the current session history to a JSON file,
    only if the history contains at least one complete turn (User + Model).

    Args:
        session_id: Unique identifier for the session
        history: Conversation history to save (universal format: List of dicts)
        system_prompt: System prompt used for the assistant
        model_name: Name of the LLM model used

    Returns:
        tuple[bool, str | None]: (success, error_message)
    """
    if len(history) < 2:
        # CONDITION: Prevents saving empty/incomplete session
        return True, None

    # Check if file already exists (might have old format name)
    existing_file = find_session_file(session_id)

    # If no existing file, generate friendly filename from first user message
    if not existing_file:
        first_user_message = None
        for entry in history:
            role = entry.get('role', '') if isinstance(entry, dict) else getattr(entry, 'role', '')
            if role == 'user':
                if isinstance(entry, dict) and 'parts' in entry and entry['parts']:
                    first_user_message = entry['parts'][0].get('text', '')
                else:
                    first_user_message = getattr(entry, 'parts', [{}])[0].get('text', '') if hasattr(entry, 'parts') else ""
                break

        if first_user_message:
            friendly_name = generate_friendly_filename(first_user_message, session_id)
            log_filename = os.path.join(LOG_DIR, f"{friendly_name}.json")
        else:
            # Fallback to old format if we can't extract first message
            log_filename = os.path.join(LOG_DIR, f"{session_id}-log.json")
    else:
        # Use existing file path
        log_filename = existing_file

    json_history = []
    for content in history:
        # Handle universal format (dictionaries) from both Gemini and LLaMA clients
        if isinstance(content, dict) and 'parts' in content and content['parts']:
            text_part = content['parts'][0].get('text', '')
        else:
            # Fallback for legacy formats
            text_part = getattr(content, 'parts', [{}])[0].get('text', '') if hasattr(content, 'parts') else ""
        
        if text_part:
            json_history.append({
                'role': content.get('role', '') if isinstance(content, dict) else getattr(content, 'role', ''),
                'timestamp': datetime.now().isoformat(),
                'text': text_part
            })

    log_data = {
        'session_id': session_id,
        'model': model_name,
        'system_role': system_prompt,
        'history': json_history
    }

    try:
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
        return True, None
        
    except Exception as e:
        return False, f"Error writing to file {log_filename}: {e}"

def list_sessions():
    """Returns a list of available sessions with their metadata."""
    files = os.listdir(LOG_DIR)

    # Collect both old format (UUID-log.json) and new format (name_shortid.json)
    session_files = []
    for f in files:
        if f.endswith('.json') and f != 'azor-wal.json':
            # Skip old format files if newer format exists
            if f.endswith('-log.json'):
                # Old format: UUID-log.json
                session_id = f.replace('-log.json', '')
                short_id = session_id[:8]
                # Check if newer format exists
                has_new_format = any(
                    fn.endswith(f"_{short_id}.json") and not fn.endswith('-log.json')
                    for fn in files
                )
                if not has_new_format:
                    session_files.append((f, session_id, None))
            else:
                # New format: friendly_name_shortid.json
                # Extract shortid (last 8 chars before .json)
                base_name = f[:-5]  # Remove .json
                if '_' in base_name:
                    parts = base_name.split('_')
                    short_id = parts[-1]
                    friendly_name = '_'.join(parts[:-1])
                    session_files.append((f, None, friendly_name))

    sessions_data = []
    for filename, old_session_id, friendly_name in sorted(session_files, key=lambda x: x[0]):
        log_path = os.path.join(LOG_DIR, filename)
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                session_id = log_data.get('session_id', old_session_id or 'unknown')
                history_len = len(log_data.get('history', []))
                last_msg_time_str = log_data.get('history', [{}])[-1].get('timestamp', 'Brak daty')

                time_str = 'Brak aktywności'
                if last_msg_time_str != 'Brak daty':
                    try:
                        dt = datetime.fromisoformat(last_msg_time_str)
                        time_str = dt.strftime('%Y-%m-%d %H:%M')
                    except ValueError:
                        pass

            # Use friendly name if available, otherwise show short session ID
            display_id = friendly_name if friendly_name else session_id[:8]

            sessions_data.append({
                'id': session_id,  # Full ID for loading
                'display_name': display_id,  # Friendly name for display
                'messages_count': history_len,
                'last_activity': time_str,
                'error': None
            })
        except Exception as e:
            display_id = friendly_name if friendly_name else (old_session_id[:8] if old_session_id else filename)
            sessions_data.append({
                'id': old_session_id or 'unknown',
                'display_name': display_id,
                'error': f'BŁĄD ODCZYTU PLIKU: {str(e)}'
            })

    return sessions_data

def remove_session_file(session_id: str) -> tuple[bool, str | None]:
    """
    Removes a session log file from the filesystem.

    Args:
        session_id: The ID of the session to remove.

    Returns:
        A tuple containing a boolean indicating success and an optional error message.
    """
    log_filename = find_session_file(session_id)
    if not log_filename:
        return False, f"Session file for ID '{session_id[:8]}...' not found."

    try:
        os.remove(log_filename)
        return True, None
    except OSError as e:
        return False, f"Error removing session file '{log_filename}': {e}"

def rename_session_file(session_id: str, new_name: str) -> tuple[bool, str | None, str | None]:
    """
    Renames a session file with a custom friendly name.

    Args:
        session_id: The ID of the session to rename
        new_name: The new friendly name for the session (will be sanitized)

    Returns:
        A tuple containing: (success, error_message, new_filename)
    """
    # Find the existing file
    old_filepath = find_session_file(session_id)
    if not old_filepath:
        return False, f"Session file for ID '{session_id[:8]}...' not found.", None

    # Sanitize the new name using the same logic as generate_friendly_filename
    sanitized_name = new_name[:50].strip()
    sanitized_name = re.sub(r'[^\w\s\-]', '', sanitized_name)
    sanitized_name = re.sub(r'\s+', '_', sanitized_name)
    sanitized_name = re.sub(r'_+', '_', sanitized_name)
    sanitized_name = sanitized_name.strip('_')

    if not sanitized_name:
        return False, "Nazwa nie może być pusta po usunięciu nieprawidłowych znaków.", None

    # Generate new filename with short ID for uniqueness
    short_id = session_id[:8]
    new_filename = f"{sanitized_name}_{short_id}.json"
    new_filepath = os.path.join(LOG_DIR, new_filename)

    # Check if target file already exists
    if os.path.exists(new_filepath) and new_filepath != old_filepath:
        return False, f"Plik o nazwie '{new_filename}' już istnieje.", None

    # Rename the file
    try:
        os.rename(old_filepath, new_filepath)
        return True, None, sanitized_name
    except OSError as e:
        return False, f"Błąd podczas zmiany nazwy pliku: {e}", None
