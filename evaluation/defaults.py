"""Repository-wide defaults for live evaluation execution."""

# Evaluation replays are deliberately real-time unless a caller explicitly
# opts into accelerated diagnostic playback.  Keeping this in one importable
# module prevents individual campaign scripts from silently reverting to max.
DEFAULT_PLAYBACK_MODE = "realtime"
DEFAULT_PLAYBACK_SPEED = 1.0
