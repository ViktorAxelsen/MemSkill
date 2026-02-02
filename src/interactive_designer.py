"""
Interactive Designer
"""
from typing import List, Any

from src.designer import Designer, DesignerCase
from prompts.designer_prompts_interactive import (
    INTERACTIVE_DESIGNER_ANALYSIS_PROMPT,
    INTERACTIVE_DESIGNER_REFLECTION_PROMPT,
    INTERACTIVE_DESIGNER_REFINEMENT_PROMPT,
)


class InteractiveDesigner(Designer):
    """Designer specialized for interactive objective-driven failures."""

    def _format_failure_cases_details(self, cases: List[DesignerCase]) -> str:
        case_details = []
        for i, case in enumerate(cases):
            case_str = f"### Case {i + 1}\n"
            case_str += f"**Objective:** {case.question}\n"
            case_str += f"**Outcome:** {case.prediction}\n"
            if case.evidence:
                case_str += "**Trajectory:**\n"
                case_str += f"{case.evidence}\n"

            if case.retrieved_memories:
                case_str += f"**Retrieved Memories ({len(case.retrieved_memories)}):**\n"
                for j, mem in enumerate(case.retrieved_memories[:20]):
                    case_str += f"  {j + 1}. {mem}\n"
            else:
                case_str += "**Retrieved Memories:** None\n"

            case_details.append(case_str)

        return "\n".join(case_details)

    def build_analysis_prompt(self, cases: List[DesignerCase], operation_bank,
                               evolution_feedback: str = "") -> str:
        cases = self._normalize_cases_for_prompt(cases)
        operation_bank_description = self._format_operation_bank_description(operation_bank)
        failure_cases_details = self._format_failure_cases_details(cases)

        new_skill_hint = ""
        if getattr(self.args, "designer_new_skill_hint", False):
            new_skill_hint = ("Note: If failures indicate a capability gap, it is encouraged to recommend "
                              "adding a new skill.")

        return INTERACTIVE_DESIGNER_ANALYSIS_PROMPT.format(
            operation_bank_description=operation_bank_description,
            evolution_feedback=evolution_feedback,
            num_failure_cases=len(cases),
            failure_cases_details=failure_cases_details,
            new_skill_hint=new_skill_hint,
            max_changes=self._get_max_changes()
        )

    def build_reflection_prompt(self, analysis_feedback: str, cases: List[Any], operation_bank,
                                 evolution_feedback: str = "", reflection_round: int = 2,
                                 reflection_round_total: int = 2) -> str:
        cases = self._normalize_cases_for_prompt(cases)
        operation_bank_description = self._format_operation_bank_description(operation_bank)
        failure_cases_details = self._format_failure_cases_details(cases)

        new_skill_hint = ""
        if getattr(self.args, "designer_new_skill_hint", False):
            new_skill_hint = ("Note: If failures indicate a capability gap, it is encouraged to recommend "
                              "adding a new skill.")

        return INTERACTIVE_DESIGNER_REFLECTION_PROMPT.format(
            analysis_feedback=analysis_feedback,
            operation_bank_description=operation_bank_description,
            evolution_feedback=evolution_feedback,
            num_failure_cases=len(cases),
            failure_cases_details=failure_cases_details,
            reflection_round=reflection_round,
            reflection_round_total=reflection_round_total,
            new_skill_hint=new_skill_hint,
            max_changes=self._get_max_changes()
        )

    def build_refinement_prompt(self, analysis_feedback: str, operation_bank,
                                 evolution_feedback: str = "") -> str:
        ops = operation_bank.get_all_operations()
        op_full_details = []
        for op in ops:
            op_detail = f"### {op.name}\n"
            op_detail += f"- **Type:** {op.update_type}\n"
            op_detail += f"- **Description:** {op.description}\n"
            op_detail += f"- **Instruction Template:**\n```\n{op.instruction_template}\n```\n"
            op_full_details.append(op_detail)

        operation_bank_full = "\n".join(op_full_details) if op_full_details else "(No evolvable operations available)"

        new_skill_hint = ""
        if getattr(self.args, "designer_new_skill_hint", False):
            new_skill_hint = ("Note: If failures indicate a capability gap, it is encouraged to recommend "
                              "adding a new skill.")

        return INTERACTIVE_DESIGNER_REFINEMENT_PROMPT.format(
            analysis_feedback=analysis_feedback,
            operation_bank_full=operation_bank_full,
            evolution_feedback=evolution_feedback,
            new_skill_hint=new_skill_hint,
            max_changes=self._get_max_changes()
        )
