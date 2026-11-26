"""Loads settings.toml, the one hand-editable file for every tunable value.

Scripts read their values from here, e.g.:
    from settings import SETTINGS
    ITERATIONS = SETTINGS["anneal"]["iterations"]

The file sits next to this script. To use another copy (e.g. a scenario), set
SETTINGS_PATH in .env to its full path.
"""
import os
from pathlib import Path

try:
    import tomllib                     # Python 3.11+
except ModuleNotFoundError:            # older Python: pip install tomli
    import tomli as tomllib

from dotenv import load_dotenv

load_dotenv()
SETTINGS_PATH = Path(os.getenv("SETTINGS_PATH") or Path(__file__).with_name("settings.toml"))

try:
    with open(SETTINGS_PATH, "rb") as f:
        SETTINGS = tomllib.load(f)
except FileNotFoundError:
    raise SystemExit(f"Settings file not found: {SETTINGS_PATH}")
except tomllib.TOMLDecodeError as e:
    raise SystemExit(f"Settings file {SETTINGS_PATH} could not be read: {e}\n"
                     "Check quotes around text and commas in lists near that line.")


def depot_list(key):
    """A depot list from [depots] (active, locked, closing), stripped of blanks."""
    return [d.strip() for d in SETTINGS["depots"].get(key, []) if d.strip()]


ACTIVE_DEPOTS = depot_list("active")
LOCKED_DEPOTS = depot_list("locked")
CLOSING_DEPOTS = depot_list("closing")
TEAM_CODES = {str(k): v for k, v in SETTINGS["depots"].get("team_codes", {}).items()}


def fleet_caps():
    """depot -> {vehicle type: vehicles per day} from [fleet]."""
    return {depot: dict(types) for depot, types in SETTINGS.get("fleet", {}).items()}


def is_active(depot):
    """True if the depot takes part in the new plan (active and not closing)."""
    if depot in CLOSING_DEPOTS:
        return False
    return not ACTIVE_DEPOTS or depot in ACTIVE_DEPOTS
