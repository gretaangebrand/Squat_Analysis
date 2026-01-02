# squat_app/sound.py
import sys

try:
    import winsound
except Exception:
    winsound = None


def play_valid_squat_sound(sound_enabled: bool) -> None:
    if not sound_enabled:
        return

    if winsound is not None:
        winsound.Beep(880, 150)
        return

    # fallback: terminal bell
    sys.stdout.write("\a")
    sys.stdout.flush()
