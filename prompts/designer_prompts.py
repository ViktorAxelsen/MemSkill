"""
Designer Prompts for Operation Evolution

Two-stage design process:
- Stage 1 (Analysis): Analyze failure cases to identify patterns and root causes
- Stage 2 (Refinement): Based on analysis, propose specific operation improvements
"""

# =============================================================================
# Stage 1: Analysis Prompt - Analyze failure cases comprehensively
# =============================================================================
DESIGNER_ANALYSIS_PROMPT = """You are an expert analyst for a memory-augmented QA system. Analyze the failure cases below to identify why the system failed and how the memory management skills should change.

## How This System Works
1. **Memory Storage**: The system applies memory management skills to decide what information to store from the text chunk.
2. **Memory Retrieval**: At question time, it retrieves the most relevant memories by semantic similarity.
3. **Answer Generation**: An LLM answers using the retrieved memories.

Failures can occur at any stage:
- **Storage failure**: Important information was never stored (skill missing or misapplied)
- **Retrieval failure**: Relevant memory exists but was not retrieved (embedding mismatch)
- **Memory quality failure**: Memory exists but is too vague or incomplete to answer

## Current Memory Management Skills
{operation_bank_description}

## Operation Evolution Feedback
{evolution_feedback}

## Failure Cases ({num_failure_cases} cases)
{failure_cases_details}

## Analysis Instructions
This is round 1 of a reflection loop. Produce a strong initial analysis that can be critiqued and improved.
1. For each case, check whether the retrieved memories contain the answer or the needed evidence.
2. If missing, decide whether it was never stored (storage failure) or stored but too weak (memory quality failure).
3. If the answer is present but not retrieved, label it retrieval failure and avoid changing skills unless the pattern repeats.
4. Group cases into patterns tied to information types, entities, temporal details, or constraints.
5. For each pattern, propose a concrete skill change: add a new skill or refine an existing one to capture missing details.
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

Focus on actionable insights. What specific change to the skill bank would prevent these failures?

Output ONLY the JSON, no other text.
"""

# =============================================================================
# Stage 1b: Reflection Prompt - Critique and improve analysis
# =============================================================================
DESIGNER_REFLECTION_PROMPT = """You are in a reflection cycle ({reflection_round}/{reflection_round_total}) for analyzing failure cases and memory management skills. Critique the previous analysis and improve it.

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
- Validate root_cause labels against the cases and retrieved memories.
- Strengthen potential_fix suggestions so they are specific and actionable.
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

# =============================================================================
# Stage 2: Refinement Prompt - Propose specific operation changes
# =============================================================================
DESIGNER_REFINEMENT_PROMPT = """Based on the failure analysis, propose a specific improvement to the memory operation bank.

## Failure Analysis (from Stage 1)
{analysis_feedback}

## Current Operation Bank
{operation_bank_full}

{evolution_feedback}

## Your Task
Propose up to {max_changes} improvements based on the analysis:

**Option A - Add New Operation**: Create a new operation if the analysis shows a capability gap (e.g., certain information types are not being captured).

**Option B - Refine Existing Operation**: Improve an existing operation's instruction template if the analysis shows it's not working well (e.g., memories are too vague, missing key details).

**Option C - No Change**: If the failures are due to retrieval issues (not operation issues), or if the current operations are already well-designed.

{new_skill_hint}

## CRITICAL Requirements
1. instruction_template MUST be a skill-style guide and MUST NOT include context placeholders (the executor injects the text chunk and retrieved memories)
2. instruction_template MUST clearly state purpose, when to use, and constraints
3. instruction_template MUST specify the allowed action type (INSERT or UPDATE only)
4. For new operations, `update_type` must be either "insert" or "update" (delete and noop operations are not evolved at this time)
5. Only propose operations with update_type "insert" or "update"
6. Avoid labels like "ENHANCED", "ADVANCED", or other marketing adjectives in descriptions or templates; keep phrasing neutral and task-specific
7. Do NOT embed output blocks; the executor handles output formatting and can apply the skill multiple times
8. The number of changes in the list MUST be <= {max_changes}
9. Do NOT modify the same operation more than once in a single response, and do NOT refine an operation you add in the same response

## Example of a Well-Designed Insert Operation
```json
{{
    "name": "extract_personal_preferences",
    "description": "Memory management skill for capturing personal preferences and habits mentioned in the text chunk.",
    "update_type": "insert",
    "instruction_template": "Skill: Insert Preferences\\nPurpose: Capture personal preferences, habits, or opinions stated in the text chunk.\\nWhen to use:\\n- The text chunk mentions likes, dislikes, routines, or goals tied to a person.\\nHow to apply:\\n- Attribute the preference to the correct person.\\n- Keep the preference specific and actionable.\\nConstraints:\\n- Avoid one-off or ambiguous statements.\\nAction type: INSERT only."
}}
```

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

## Instruction Template Structure
When writing instruction templates, follow this structure:

```
Skill: [Short skill name]
Purpose: [What this skill does]
When to use:
- [Trigger 1]
- [Trigger 2]
How to apply:
- [Step 1]
- [Step 2]
Constraints:
- [What to avoid]
Action type: [INSERT only | UPDATE only]
```

Output ONLY the JSON, no other text.
"""
