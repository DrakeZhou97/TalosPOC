from enum import Enum


class AdmittanceState(str, Enum):
    YES = "yes"
    NO = "no"
    MORE = "more_info_needed"
