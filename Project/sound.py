# squat_app/sound.py
import sys

# Versuche, winsound für Windows zu importieren (vermeidet ImportError bei Mac/Linux)
try:
    import winsound
except Exception:
    winsound = None

# Funktion, um einen Ton für einen validen Squat zu spielen
def play_valid_squat_sound(sound_enabled: bool) -> None:
    if not sound_enabled:
        return

    # Windows-spezifischer Ton
    if winsound is not None:
        winsound.Beep(880, 150) #880 Hz für 150 ms
        return

    # ASCII "Bell"-Ton
    sys.stdout.write("\a")
    sys.stdout.flush()
