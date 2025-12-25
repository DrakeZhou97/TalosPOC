![Banner](./static/talos.png)

Lab Assistance Proof of Concept version

Talos is an intelligent laboratory assistant robot designed for use in small molecule synthesis and DMPK (Drug Metabolism and Pharmacokinetics) labs. Talos serves as a conversational agent, streamlining lab workflows and supporting scientists throughout complex laboratory processes.

- **Automated Task Delegation:** Talos can assign and coordinate various laboratory procedures, including TLC (Thin Layer Chromatography) spotting, column chromatography, LC-MS sample preparation and submission, rotary evaporation, weighing, and inventory tracking.
- **Chemistry Domain Q&A:** The assistant helps answer chemistry-related questions, providing recommendations for TLC conditions, column and rotary evaporation parameters, and offering compound property lookups per user requests.
- **Lab Operations Insights:** Talos enables users to query experiment task progress, monitor the status of robots and instruments, and check material locations and inventory states in real time.

Talos is purpose-built for small molecule synthesis and DMPK laboratory applications. Requests that fall outside this domain are not supported.


## Task Decomposition / Plan Generation

Generate the task planning (TODO List) based on Messages. Once the user confirms the plan, the system automatically forwards it to the corresponding executor for execution. When updating the state machine, confirm the current execution status using `plan_cursor` and `plan` in TLCState.

Following are system pre-defined executor, check `class ExecutorKey(StrEnum)`

```python
class ExecutorKey(StrEnum):
    TLC_AGENT = "tlc_agent.run"
    # COLUMN_AGENT = "column.recommend"
    # ROBOT_TLC = "robot.tlc_spot"
    # PROPERTY_LOOKUP = "property.lookup"
```

### TLC Agent

Once entering the TLC section, TLC Agent needs to fill a form in collaboration with the user. The form primarily collects compound name and molecular formula, which is transmitted to the frontend for rendering. Users can manually edit the form content on the frontend or use natural language descriptions to allow the Agent to automatically assist with modifications. This multi-turn conversation continues until the user confirms the form.

When compounds are confirmed, it will call TLC MCP server to recommend ratio of develop solvent.


## Enginerring Architecture

`functions` folder contains Nodes / Agent that acts as coordinator which controls the flow going. `agents/` contains exectuor agent that coorrespondent to actual Chem technics. `run()` functions under each Agent is the Entry point

`agent_mapper.py` is a mapper class which wrap Agents as Langgraph functionable runnable node. Only things related to Langgraph or minimum logic should be written under there. 



### Tech Stack

1. Langchain + Langgraph 1.0
2. Pydantic V2
3. Python 3.12
4. ruff
5. loguru
