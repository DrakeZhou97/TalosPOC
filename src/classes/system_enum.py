from enum import Enum


class AdmittanceState(str, Enum):
    YES = "yes"
    NO = "no"
    MORE = "more_info_needed"


class GoalTypeEnum(str, Enum):
    CONSULTING = "consulting"
    EXECUTION = "execution"
    MANAGEMENT = "management"


class ExecutionStatusEnum(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"


class TLCPhase(str, Enum):
    COLLECTING = "collecting"
    AWAITING_INFO = "awaiting_info"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    DONE = "done"
