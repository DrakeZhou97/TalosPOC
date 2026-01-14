from enum import Enum, StrEnum


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


class ExecutorKey(StrEnum):
    TLC_AGENT = "tlc_agent"
    CC_AGENT = "cc_agent"
    RE_AGENT = "re_agent"
    # COLUMN_AGENT = "column.recommend"
    # ROBOT_TLC = "robot.tlc_spot"
    # PROPERTY_LOOKUP = "property.lookup"


class TLCPhase(str, Enum):
    COLLECTING = "collecting"
    AWAITING_INFO = "awaiting_info"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    DONE = "done"

class CCPhase(str, Enum):
    COLLECTING = "collecting"
    EQUIPMENT_VALIDATED = "equipment_validated"
    SPEC_CONFIRMED = "spec_confirmed"
    PARAMS_CONFIRMED = "params_confirmed"
    DONE = "done"

class REPhase(str, Enum):
    COLLECTING = "collecting"
    EQUIPMENT_VALIDATED = "equipment_validated"
    SPEC_CONFIRMED = "spec_confirmed"
    PARAMS_CONFIRMED = "params_confirmed"
    DONE = "done"