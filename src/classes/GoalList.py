from pydantic import BaseModel, Field


class GoalTemplate(BaseModel):
    id: str = Field(..., description="Unique identifier for the task template.")
    type: str = Field(..., description="Type of the template.")
    name: str = Field(..., description="Name of the task template.")
    description: str = Field(..., description="Detailed description of the task template.")
    spec_sheet_ids: list[str] = Field(..., description="List of associated fact sheets.")


CapabilityCargo: list[GoalTemplate] = [
    GoalTemplate(
        id="task_template_001",
        type="safety_assessment",
        name="Chemical Safety Assessment",
        description="Assess the safety protocols for handling chemicals in the laboratory.",
        spec_sheet_ids=["fact_sheet_chem_safety_001", "fact_sheet_chem_safety_002"],
    ),
    GoalTemplate(
        id="task_template_002",
        type="equipment_maintenance",
        name="Equipment Maintenance Schedule",
        description="Create a maintenance schedule for laboratory equipment to ensure optimal performance.",
        spec_sheet_ids=["fact_sheet_equip_maint_001"],
    ),
    GoalTemplate(
        id="task_template_003",
        type="tlc_procedures",
        name="TLC Procedures",
        description="Design the ratio of material and solvent for Thin Layer Chromatography (TLC) experiments.",
        spec_sheet_ids=["fact_sheet_tlc_001"],
    ),
]
