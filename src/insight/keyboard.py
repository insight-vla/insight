"""Keyboard early-stop listener (env-agnostic).

Long-running plan-execute loops want a "press s/q to abort gracefully" signal
that doesn't require Ctrl-C (which kills mid-write and corrupts artifacts).

Usage:
    from insight.keyboard import start_stop_listener, stop_requested
    start_stop_listener()
    while not stop_requested():
        ...

Process-wide singleton — calling ``start_stop_listener`` twice is a no-op so
callers don't have to coordinate.
"""

from __future__ import annotations

import atexit
import logging
import select
import sys
import threading


_stop_event = threading.Event()
_listener_started = False
_saved_termios: tuple[int, list] | None = None


def _restore_terminal() -> None:
    """Restore the terminal to its pre-cbreak settings.

    Registered with ``atexit`` so it fires on normal exit AND on a
    KeyboardInterrupt-induced exit. The daemon listener thread's own
    ``try/finally`` cleanup doesn't run on Ctrl+C because daemon threads
    get killed without their finally clauses executing — this hook is the
    safety net so the user's terminal isn't left in cbreak mode.
    """
    global _saved_termios
    if _saved_termios is None:
        return
    try:
        import termios
        fd, settings = _saved_termios
        termios.tcsetattr(fd, termios.TCSADRAIN, settings)
    except Exception:
        pass
    finally:
        _saved_termios = None


def start_stop_listener() -> None:
    """Start a daemon thread that sets the stop flag on 's' or 'q' keypress.

    No-op if already started. Silently exits on platforms where termios isn't
    available (e.g. Windows) — callers using ``stop_requested`` will simply
    never see the flag toggle, which is fine for non-interactive runs.
    """
    global _listener_started, _saved_termios
    if _listener_started:
        return
    _listener_started = True

    # Snapshot terminal settings on the main thread (where we know stdin is a
    # tty) and register the restore hook before spawning the listener. atexit
    # runs even on KeyboardInterrupt, so the terminal always gets restored —
    # whereas the daemon thread's own finally would be skipped on Ctrl+C.
    try:
        import termios
        fd = sys.stdin.fileno()
        _saved_termios = (fd, termios.tcgetattr(fd))
        atexit.register(_restore_terminal)
    except Exception:
        # No tty (e.g. piped input, Windows) — listener will silently no-op.
        return

    def _listener() -> None:
        try:
            import termios
            import tty
            fd, _ = _saved_termios  # type: ignore[misc]
            tty.setcbreak(fd)
            while not _stop_event.is_set():
                if select.select([sys.stdin], [], [], 0.5)[0]:
                    ch = sys.stdin.read(1)
                    if ch.lower() in ("s", "q"):
                        _stop_event.set()
                        logging.info(
                            "\n*** STOP REQUESTED (press detected) — saving and exiting... ***"
                        )
                        break
            # Normal exit path: restore now so we don't have to wait for atexit.
            _restore_terminal()
        except Exception:
            pass

    t = threading.Thread(target=_listener, daemon=True)
    t.start()


def stop_requested() -> bool:
    """Return True if 's'/'q' was pressed (or ``request_stop`` was called)."""
    return _stop_event.is_set()


def request_stop() -> None:
    """Set the stop flag programmatically (e.g. from a signal handler)."""
    _stop_event.set()


def reset_stop() -> None:
    """Clear the stop flag — useful when reusing the listener across runs."""
    _stop_event.clear()
