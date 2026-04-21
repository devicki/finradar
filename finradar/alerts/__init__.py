"""Breaking-news alerts — push significant articles to external channels.

Phase 3 Day 3: Discord webhook only. Telegram + email land in Phase 4 when
user preferences / auth exist.
"""

from finradar.alerts.dispatcher import (
    AlertTrigger,
    dispatch_pending_alerts,
    evaluate_trigger,
)

__all__ = [
    "AlertTrigger",
    "dispatch_pending_alerts",
    "evaluate_trigger",
]
