"""
Designer prompts for interactive ALFWorld-style training.
"""

INTERACTIVE_DESIGNER_ANALYSIS_PROMPT = """You are an expert analyst for a memory-augmented interactive agent. Analyze failure cases to determine how memory management skills should change.

## How This System Works
1. **Memory Storage**: The system applies memory management skills to trajectory text to store useful experience.
2. **Memory Retrieval**: At action time, it retrieves relevant memories using the task objective as a query.
3. **Action Selection**: An LLM chooses actions using retrieved memories and the accumulated interaction context.

## Task Context (Why Memory Matters)
- Text-only, partially observable environment: the agent only sees current observations and must infer state changes across steps.
- Long-horizon, multi-step goals (e.g., find, take, clean/heat/cool, place, examine).
- Actions have preconditions and state constraints: containers must be open, objects must be held, devices must be on/off, and locations must be reached before interacting.
- Observations reveal object locations and states; inventory is critical evidence.
- Skills are applied to trajectories, so they should capture reusable procedures, preconditions, and state transitions without hard-coding instance IDs.

Failures can occur at any stage:
- **Storage failure**: Important procedures or constraints were never stored (skill missing or misapplied).
- **Retrieval failure**: Relevant memory exists but was not retrieved for the objective.
- **Memory quality failure**: Memory exists but is too vague, missing key steps, or not actionable.

## Current Memory Management Skills
{operation_bank_description}

## Operation Evolution Feedback
{evolution_feedback}

## Failure Cases ({num_failure_cases} cases)
{failure_cases_details}

## Analysis Instructions
This is round 1 of a reflection loop. Produce a strong initial analysis that can be critiqued and improved.
1. For each case, check whether retrieved memories contain the missing procedure, constraints, or key objects.
2. If missing, decide whether it was never stored (storage failure) or stored but too weak (memory quality failure).
3. If the memory is present but not retrieved, label it retrieval failure and avoid changing skills unless the pattern repeats.
4. Group cases into patterns tied to objectives, object types, temporal steps, state transitions, or action constraints.
5. For each pattern, propose a concrete skill change: add a new skill or refine an existing one.
6. Provide up to {max_changes} recommendations total (use fewer if only one change is justified).
{new_skill_hint}

## Output Format
Provide your analysis as JSON:
{{
    "failure_patterns": [
        {{
            "pattern_name": "<descriptive name for this failure pattern>",
            "affected_cases": [<list of case numbers, e.g., 1, 3, 5>],
            "root_cause": "<storage_failure|retrieval_failure|memory_quality_failure>",
            "explanation": "<why this pattern of failures is occurring>",
            "potential_fix": "<what kind of operation change could address this>"
        }}
    ],
    "recommendations": [
        {{
            "action": "<add_new_operation|refine_existing_operation|no_change>",
            "target_operation": "<operation name to refine, or null if adding new>",
            "rationale": "<clear explanation of why this is the best improvement>",
            "priority": "<high|medium|low>"
        }}
    ],
    "summary": "<1-2 sentence summary of main findings>"
}}

Output ONLY the JSON, no other text.
"""


INTERACTIVE_DESIGNER_REFLECTION_PROMPT = """You are in a reflection cycle ({reflection_round}/{reflection_round_total}) for analyzing interactive failure cases. Critique the previous analysis and improve it.

## Previous Analysis (from prior round)
{analysis_feedback}

## Current Memory Management Skills
{operation_bank_description}

## Operation Evolution Feedback
{evolution_feedback}

## Failure Cases ({num_failure_cases} cases)
{failure_cases_details}

## Reflection Instructions
- Check for missing or misclassified failure patterns.
- Validate root_cause labels against the objectives and retrieved memories.
- Strengthen potential_fix suggestions so they are specific and actionable, especially for preconditions, object states, and reusable subgoals.
- Keep the same output format and output only JSON.
- Provide up to {max_changes} recommendations total (use fewer if only one change is justified).
{new_skill_hint}

## Output Format
Provide your analysis as JSON:
{{
    "failure_patterns": [
        {{
            "pattern_name": "<descriptive name for this failure pattern>",
            "affected_cases": [<list of case numbers, e.g., 1, 3, 5>],
            "root_cause": "<storage_failure|retrieval_failure|memory_quality_failure>",
            "explanation": "<why this pattern of failures is occurring>",
            "potential_fix": "<what kind of operation change could address this>"
        }}
    ],
    "recommendations": [
        {{
            "action": "<add_new_operation|refine_existing_operation|no_change>",
            "target_operation": "<operation name to refine, or null if adding new>",
            "rationale": "<clear explanation of why this is the best improvement>",
            "priority": "<high|medium|low>"
        }}
    ],
    "summary": "<1-2 sentence summary of main findings>"
}}

Output ONLY the JSON, no other text.
"""


INTERACTIVE_DESIGNER_REFINEMENT_PROMPT = """Based on the failure analysis, propose a specific improvement to the memory operation bank for interactive tasks.

## Failure Analysis (from Stage 1)
{analysis_feedback}

## Current Operation Bank
{operation_bank_full}

{evolution_feedback}

## Your Task
Propose up to {max_changes} improvements based on the analysis:

**Option A - Add New Operation**: Create a new operation if the analysis shows a capability gap.
**Option B - Refine Existing Operation**: Improve an existing operation if memories are too vague or miss key steps.
**Option C - No Change**: If failures are due to retrieval issues or the operations are already well-designed.

{new_skill_hint}

## CRITICAL Requirements
1. instruction_template MUST be a skill-style guide and MUST NOT include context placeholders.
2. instruction_template MUST clearly state purpose, when to use, and constraints.
3. instruction_template MUST specify the allowed action type (INSERT or UPDATE only).
4. For new operations, update_type must be "insert" or "update".
5. Avoid marketing adjectives; keep phrasing neutral and task-specific.
6. Do NOT embed output blocks; the executor handles output formatting.
7. Templates should generalize across task instances; avoid hard-coding object IDs or exact room names.
8. The number of changes MUST be <= {max_changes}.
9. Do NOT modify the same operation more than once in a single response.

## Output Format
Respond with ONE of these JSON structures:

### One or more changes (up to {max_changes}):
{{
    "action": "apply_changes",
    "summary": "<overall rationale for the set of changes>",
    "changes": [
        {{
            "action": "add_new",
            "new_operation": {{
                "name": "<snake_case_name>",
                "description": "<what it does and when to use it>",
                "instruction_template": "<skill-style instruction template>",
                "update_type": "<insert|update>",
                "reasoning": "<how this addresses the identified failures>"
            }}
        }},
        {{
            "action": "refine_existing",
            "refined_operation": {{
                "name": "<existing_operation_name>",
                "changes": {{
                    "description": "<improved description>",
                    "instruction_template": "<improved template>"
                }},
                "reasoning": "<how these changes address the identified failures>"
            }}
        }}
    ]
}}

### No changes needed:
{{
    "action": "no_change",
    "reasoning": "<why the current operations are sufficient>"
}}

Output ONLY the JSON, no other text.
"""
