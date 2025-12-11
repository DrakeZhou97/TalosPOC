from typing import Literal


class Survey:
    def is_finished(self, survey_obj: dict) -> Literal["PROCEED", "REVISE"]:
        """
        Checkpoint function determine if we clarify human intention and ready for execution.

        We suppose to have a state that store the survey filled status, after each round.
        This function is to check if the survey is finished or not, which means user's intention is clarified.
        """
        if survey_obj.get("FINISHED", False):
            return "PROCEED"

        return "REVISE"
