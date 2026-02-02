"""
Initial Operation Templates (Seed Operations)
"""

# Memory management skill descriptions and instructions
SKILL_INSERT_DESCRIPTION = (
    "Memory management skill for capturing new, durable facts from the current text chunk that "
    "are not already in memory."
)
SKILL_INSERT_TEMPLATE = """Skill: Insert New Memory
Purpose: Capture new, durable facts from the current text chunk that are missing in memory.
When to use:
- The text chunk introduces new facts, events, plans, or context worth storing.
- The information is stable and likely useful later.
How to apply:
- Compare against retrieved memories to avoid duplicates.
- Split distinct facts into separate items.
- Keep each item concise and specific.
Constraints:
- Skip trivial, fleeting, or speculative content.
- Do not update or delete existing memories.
Action type: INSERT only.
"""

SKILL_UPDATE_DESCRIPTION = (
    "Memory management skill for revising an existing memory item when the text chunk provides "
    "corrections or new details."
)
SKILL_UPDATE_TEMPLATE = """Skill: Update Existing Memory
Purpose: Revise a retrieved memory with new or corrected information from the text chunk.
When to use:
- The text chunk clarifies, corrects, or extends a retrieved memory.
How to apply:
- Select the best matching memory item.
- Merge new details into a single updated item.
- Preserve accurate details that still hold.
Constraints:
- Do not create new memories.
- Do not delete items.
Action type: UPDATE only.
"""

SKILL_DELETE_DESCRIPTION = (
    "Memory management skill for removing memory items that are incorrect, outdated, "
    "or superseded."
)
SKILL_DELETE_TEMPLATE = """Skill: Delete Invalid Memory
Purpose: Remove a retrieved memory that is wrong, outdated, or superseded by the text chunk.
When to use:
- The text chunk clearly contradicts a memory.
- A plan or fact is explicitly canceled or replaced.
How to apply:
- Only delete when evidence is explicit.
Constraints:
- If uncertain, prefer no action over deletion.
Action type: DELETE only.
"""

SKILL_NOOP_DESCRIPTION = (
    "Memory management skill for confirming that no memory changes are required."
)
SKILL_NOOP_TEMPLATE = """Skill: No Operation
Purpose: Confirm no memory changes are needed for the text chunk.
When to use:
- The text chunk contains no new, corrective, or actionable information.
Constraints:
- Emit NOOP only if none of the selected skills produce actions.
Action type: NOOP only.
"""

INITIAL_OPERATIONS = {
    "insert": {
        "name": "insert",
        "description": SKILL_INSERT_DESCRIPTION,
        "instruction_template": SKILL_INSERT_TEMPLATE,
        "update_type": "insert",  # insert, update, delete, noop
        "meta_info": {
            "usage_count": 0,
            "avg_reward": 0.0,
            "recent_rewards": [],
            "recent_usage_ema": 0.0,
            "created_at": "initial",
            "last_modified": "initial"
        }
    },

    "update": {
        "name": "update",
        "description": SKILL_UPDATE_DESCRIPTION,
        "instruction_template": SKILL_UPDATE_TEMPLATE,
        "update_type": "update",
        "meta_info": {
            "usage_count": 0,
            "avg_reward": 0.0,
            "recent_rewards": [],
            "recent_usage_ema": 0.0,
            "created_at": "initial",
            "last_modified": "initial"
        }
    },

    "delete": {
        "name": "delete",
        "description": SKILL_DELETE_DESCRIPTION,
        "instruction_template": SKILL_DELETE_TEMPLATE,
        "update_type": "delete",
        "meta_info": {
            "usage_count": 0,
            "avg_reward": 0.0,
            "recent_rewards": [],
            "recent_usage_ema": 0.0,
            "created_at": "initial",
            "last_modified": "initial"
        }
    },

    "noop": {
        "name": "noop",
        "description": SKILL_NOOP_DESCRIPTION,
        "instruction_template": SKILL_NOOP_TEMPLATE,
        "update_type": "noop",
        "meta_info": {
            "usage_count": 0,
            "avg_reward": 0.0,
            "recent_rewards": [],
            "recent_usage_ema": 0.0,
            "created_at": "initial",
            "last_modified": "initial"
        }
    }
}

def get_initial_operations(include_noop: bool = False):
    """Return a deep copy of initial operations, optionally including noop."""
    import copy
    ops = copy.deepcopy(INITIAL_OPERATIONS)
    if not include_noop:
        ops.pop("noop", None)
    return ops
