from files import session_files
from cli import console


def rename_session_command(session_id: str, new_name: str):
    """
    Renames a session file with a custom friendly name.

    Args:
        session_id: The ID of the session to rename (can be full UUID or short ID)
        new_name: The new friendly name for the session
    """
    if not new_name or not new_name.strip():
        console.print_error("Błąd: Nowa nazwa nie może być pusta.")
        return

    # Get all sessions to find matching session by ID or display name
    sessions = session_files.list_sessions()

    # Try to find the session
    target_session_id = None
    for session in sessions:
        # Match by full ID, short ID, or display name
        if (session['id'] == session_id or
            session['id'].startswith(session_id) or
            session.get('display_name', '') == session_id):
            target_session_id = session['id']
            break

    if not target_session_id:
        console.print_error(f"Błąd: Nie znaleziono sesji o ID lub nazwie: {session_id}")
        console.print_info("Użyj /session list aby zobaczyć dostępne sesje.")
        return

    # Get current display name
    old_display_name = session_files.get_session_display_name(target_session_id)

    # Rename the session
    success, error, sanitized_name = session_files.rename_session_file(target_session_id, new_name)

    if success:
        console.print_info(f"\n✓ Zmieniono nazwę sesji:")
        console.print_info(f"  Stara nazwa: {old_display_name}")
        console.print_info(f"  Nowa nazwa:  {sanitized_name}")
        console.print_info(f"  ID sesji:    {target_session_id[:8]}...")
    else:
        console.print_error(f"Błąd podczas zmiany nazwy sesji: {error}")
