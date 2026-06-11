"""Moduli condivisi: path singleton, dispatcher base, schema migration, hb/queue lib."""

from common.dispatcher_base import (
    BaseDispatcher,
    DispatcherProtocol,
    validate_dispatch_key,
)
from common.hb_lib import read_heartbeat, write_heartbeat
from common.paths import BIDS_ROOT, DERIV, SUBJECTS_DIR
from common.queue_lib import append_sprint, queue_lock, update_status
from common.state_lib import (
    apply_migration,
    load_schema_version,
    migrate_on_load,
    register_migration,
    rollback_migration,
)

__all__ = [
    "BaseDispatcher",
    "DispatcherProtocol",
    "validate_dispatch_key",
    "read_heartbeat",
    "write_heartbeat",
    "BIDS_ROOT",
    "DERIV",
    "SUBJECTS_DIR",
    "append_sprint",
    "queue_lock",
    "update_status",
    "apply_migration",
    "load_schema_version",
    "migrate_on_load",
    "register_migration",
    "rollback_migration",
]
