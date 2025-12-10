from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Optional

WEEKDAY_KEYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def build_daily_availability(
    start_date: date,
    end_date: date,
    weekly_hours: Dict[str, float],
    adhoc_hours: Optional[Dict[str, float]] = None,
    daily_max_hours: float = 8.0,
) -> Dict[date, float]:
    adhoc_hours = adhoc_hours or {}
    availability: Dict[date, float] = {}
    current = start_date
    while current <= end_date:
        weekday_key = WEEKDAY_KEYS[current.weekday()]
        base_hours = weekly_hours.get(weekday_key, 0.0)
        date_key = current.isoformat()
        hours = adhoc_hours.get(date_key, base_hours)
        availability[current] = min(hours, daily_max_hours)
        current += timedelta(days=1)
    return availability
