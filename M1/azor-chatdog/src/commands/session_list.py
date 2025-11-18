from files import session_files
from cli import console

def list_sessions_command():
    """Displays a formatted list of available sessions."""
    sessions = session_files.list_sessions()
    if sessions:
        console.print_help("\n--- Dostępne zapisane sesje ---")
        for session in sessions:
            if session.get('error'):
                console.print_error(f"- {session.get('display_name', session['id'][:8])} ({session['error']})")
            else:
                display_name = session.get('display_name', session['id'][:8])
                console.print_help(f"- {display_name} (Wiadomości: {session['messages_count']}, Ost. aktywność: {session['last_activity']})")
        console.print_help("------------------------------------")
        console.print_help("Aby załadować sesję, użyj komendy /load <pełne-ID-sesji>")
    else:
        console.print_help("\nBrak zapisanych sesji.")
