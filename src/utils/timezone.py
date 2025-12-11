"""Timezone utilities for the application."""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_app_timezone() -> ZoneInfo:
    """Get the application timezone from environment variable.
    
    Returns:
        ZoneInfo: The timezone object for the application.
        
    Raises:
        ValueError: If the timezone is invalid.
        
    """
    tz_name = os.getenv("TZ", "UTC")
    try:
        return ZoneInfo(tz_name)
    except Exception as exc:
        raise ValueError(f"Invalid timezone '{tz_name}' in TZ environment variable") from exc


def now() -> datetime:
    """Get the current datetime in the application timezone.
    
    Returns:
        datetime: Current datetime with application timezone.
        
    """
    return datetime.now(get_app_timezone())


# Application timezone instance
APP_TIMEZONE = get_app_timezone()
