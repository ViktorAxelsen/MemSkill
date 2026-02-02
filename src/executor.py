"""
Executor: Executes operations using LLM API
"""
import json
import re
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from json_repair import repair_json
from llm_utils import get_llm_response_via_api
from rag_utils import get_embeddings
from typing import List, Union


class ExecutionResult:
    """Result of operation execution"""
    def __init__(self, action_type: str, success: bool,
                 memory_index: int = -1, memory_content: str = "",
                 reasoning: str = ""):
        self.action_type = action_type  # INSERT, UPDATE, DELETE, NOOP
        self.success = success
        self.memory_index = memory_index
        self.memory_content = memory_content
        self.reasoning = reasoning

    def __repr__(self):
        return f"ExecutionResult(action={self.action_type}, success={self.success}, " \
               f"mem_idx={self.memory_index}, reasoning={self.reasoning[:100]}...)"


class Executor:
    """
    Executor executes operations by calling LLM API
    Parses LLM output and applies changes to memory bank
    """
    def __init__(self, args):
        self.args = args
        self.retriever_name = args.retriever
        self.logger = logging.getLogger('AgenticMemory')

    def _build_executor_prompt(self, operations: List, session_text: str,
                               retrieved_memories: List[str]) -> str:
        """Build the executor prompt from selected memory management skills."""
        if len(retrieved_memories) > 0:
            mem_text = "\n".join([f"{i}. {mem}" for i, mem in enumerate(retrieved_memories)])
        else:
            mem_text = "(No existing memories retrieved)"

        skill_blocks = []
        seen = set()
        for op in operations:
            name = getattr(op, "name", None)
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)

            description = str(getattr(op, "description", "")).strip()
            instructions = str(getattr(op, "instruction_template", "")).strip()
            update_type = str(getattr(op, "update_type", "")).strip().upper()

            lines = [f"[Skill {len(seen)}] {name}"]
            if description:
                lines.append(f"Description: {description}")
            if update_type:
                lines.append(f"Allowed action: {update_type}")
            if instructions:
                lines.append("Instructions:")
                lines.append(instructions)
            skill_blocks.append("\n".join(lines))

        skills_text = "\n\n".join(skill_blocks) if skill_blocks else "(No skills provided)"

        return (
            "You are a memory management executor. Apply the selected skills to the input text\n"
            "chunk and retrieved memories, then output memory actions.\n\n"
            "Input Text Chunk:\n"
            f"{session_text}\n\n"
            "Retrieved Memories (0-based index):\n"
            f"{mem_text}\n\n"
            "Selected Skills:\n"
            f"{skills_text}\n\n"
            "Guidelines:\n"
            "- Apply any skill as needed; a skill may be used multiple times.\n"
            "- Read the input text chunk carefully line by line and apply any skill as needed.\n"
            "- Only use action types supported by the selected skills.\n"
            "- MEMORY_INDEX is 0-based and must reference the retrieved memories list.\n"
            "- Output only action blocks in the format below.\n"
            "- Do not include explanations or REASONING lines.\n"
            "Output format (repeat as needed). Use ONE block per action and separate blocks with"
            " a blank line:\n\n"
            "INSERT block:\n"
            "ACTION: INSERT\n"
            "MEMORY_ITEM: <concise but complete summary with essential details>\n\n"
            "UPDATE block:\n"
            "ACTION: UPDATE\n"
            "MEMORY_INDEX: <0-based index>\n"
            "UPDATED_MEMORY: <concise but complete merged summary with essential updates>\n\n"
            "DELETE block:\n"
            "ACTION: DELETE\n"
            "MEMORY_INDEX: <0-based index>\n\n"
        )

    def execute_operation(self, operation: Union[object, List[object]], session_text: str,
                          retrieved_memories: List[str]) -> List[ExecutionResult]:
        """
        Execute a memory operation
        Args:
            operation: Operation object or list of Operation objects (skills)
            session_text: current session text
            retrieved_memories: list of retrieved memory contents
        Returns:
            List[ExecutionResult]: list of execution results (LLM may return multiple actions)
        """
        if isinstance(operation, (list, tuple)):
            operations = list(operation)
        else:
            operations = [operation]

        operations = [op for op in operations if op is not None]
        if len(operations) == 0:
            return [ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning="No operations provided"
            )]

        sub_chunks = [session_text]
        all_results = []
        for sub_text in sub_chunks:
            if not sub_text.strip():
                continue
            # Build executor prompt from selected skills
            instruction = self._build_executor_prompt(operations, sub_text, retrieved_memories)

            # Call LLM API
            try:
                  response, _, _ = get_llm_response_via_api(
                      prompt=instruction,
                      LLM_MODEL=self.args.model,
                      base_url=self.args.api_base,
                      api_key=self.args.api_key,
                      MAX_TOKENS=self.args.max_new_tokens,
                      TAU=self.args.temperature,
                      MAX_TRIALS=10,
                      TIME_GAP=3,
                  )
            except Exception as e:
                self.logger.warning(f"Executor API call failed: {e}")
                all_results.append(ExecutionResult(
                    action_type="NOOP",
                    success=False,
                    reasoning=f"API call failed: {str(e)}"
                ))
                continue

            # Parse response (may contain multiple actions)
            all_results.extend(self._parse_response(response, len(retrieved_memories)))

        if not all_results:
            return [ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning="No executor results produced"
            )]
        return all_results

    def _parse_response(self, response: str, num_retrieved: int) -> List[ExecutionResult]:
        """
        Parse LLM response to extract actions and content.
        Supports multiple ACTION blocks in a single response.

        Expected format (can repeat multiple times):
        ACTION: INSERT/UPDATE/DELETE/NOOP
        [MEMORY_INDEX: <index>]  # for UPDATE/DELETE
        [MEMORY_ITEM/UPDATED_MEMORY: <content>]  # for INSERT/UPDATE
        REASONING: <reasoning>

        Returns:
            List[ExecutionResult]: list of parsed results
        """
        response = self._normalize_response(response)
        results = []

        # Split response into individual action blocks
        # Find all positions where "ACTION:" appears
        action_pattern = re.compile(
            r'(?<!\w)ACTION\s*(?::|=|-)?\s*(INSERT|UPDATE|DELETE|NOOP)\b',
            re.IGNORECASE
        )
        action_matches = list(action_pattern.finditer(response))

        if not action_matches:
            json_results = self._parse_json_response(response, num_retrieved)
            if json_results:
                return json_results
            self.logger.warning("ACTION: INSERT/UPDATE/DELETE/NOOP PARSE FAILED")
            print(response)
            return [ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning="Failed to parse ACTION from response"
            )]

        # Parse each action block
        for i, match in enumerate(action_matches):
            # Determine the block boundaries
            block_start = match.start()
            block_end = action_matches[i + 1].start() if i + 1 < len(action_matches) else len(response)
            block = response[block_start:block_end].strip()

            result = self._parse_single_action(block, num_retrieved)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)

        return results

    def _normalize_response(self, response: str) -> str:
        text = response.replace("\r\n", "\n").strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if "\n" in text:
                    first_line, rest = text.split("\n", 1)
                    if first_line.strip().lower() in ("json", "text"):
                        text = rest
        return text.strip()

    def _parse_json_response(self, response: str, num_retrieved: int) -> List[ExecutionResult]:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            return []
        try:
            json_str = response[json_start:json_end]
            repaired_json = repair_json(json_str)
            data = json.loads(repaired_json)
        except Exception:
            return []

        items = []
        if isinstance(data, dict):
            if isinstance(data.get("actions"), list):
                items = data["actions"]
            else:
                items = [data]
        elif isinstance(data, list):
            items = data
        else:
            return []

        results = []
        for item in items:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", item.get("ACTION", ""))).strip().upper()
            if action == "INSERT":
                content = str(item.get("memory_item", item.get("MEMORY_ITEM", ""))).strip()
                if not content:
                    continue
                results.append(ExecutionResult(
                    action_type="INSERT",
                    success=True,
                    memory_content=content,
                    reasoning=str(item.get("reasoning", "")).strip() or "No reasoning provided"
                ))
            elif action == "UPDATE":
                idx = item.get("memory_index", item.get("MEMORY_INDEX", None))
                content = str(item.get("updated_memory", item.get("UPDATED_MEMORY", ""))).strip()
                if idx is None or content == "":
                    continue
                try:
                    idx = int(idx)
                except Exception:
                    continue
                if idx < 0 or idx >= num_retrieved:
                    continue
                results.append(ExecutionResult(
                    action_type="UPDATE",
                    success=True,
                    memory_index=idx,
                    memory_content=content,
                    reasoning=str(item.get("reasoning", "")).strip() or "No reasoning provided"
                ))
            elif action == "DELETE":
                idx = item.get("memory_index", item.get("MEMORY_INDEX", None))
                if idx is None:
                    continue
                try:
                    idx = int(idx)
                except Exception:
                    continue
                if idx < 0 or idx >= num_retrieved:
                    continue
                results.append(ExecutionResult(
                    action_type="DELETE",
                    success=True,
                    memory_index=idx,
                    reasoning=str(item.get("reasoning", "")).strip() or "No reasoning provided"
                ))
            elif action == "NOOP":
                results.append(ExecutionResult(
                    action_type="NOOP",
                    success=True,
                    reasoning=str(item.get("reasoning", "")).strip() or "No reasoning provided"
                ))

        return results

    def _parse_single_action(self, block: str, num_retrieved: int) -> ExecutionResult:
        """
        Parse a single action block.

        Args:
            block: text containing one ACTION block
            num_retrieved: number of retrieved memories (for index validation)
        Returns:
            ExecutionResult
        """
        # Extract ACTION
        action_match = re.search(
            r'ACTION\s*(?::|=|-)?\s*(INSERT|UPDATE|DELETE|NOOP)\b',
            block,
            re.IGNORECASE
        )
        if not action_match:
            return ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning="Failed to parse ACTION from block"
            )

        action_type = action_match.group(1).upper()

        # Extract REASONING
        reasoning_match = re.search(
            r'REASONING\s*(?::|=|-)?\s*(.+?)(?=[\s,;]*(?:ACTION\s*(?::|=|-)?|MEMORY[_ ]ITEM\s*(?::|=|-)?|'
            r'UPDATED[_ ]MEMORY\s*(?::|=|-)?|MEMORY[_ ]INDEX\s*(?::|=|-)?|$))',
            block,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

        # Parse based on action type
        if action_type == "NOOP":
            return ExecutionResult(
                action_type="NOOP",
                success=True,
                reasoning=reasoning
            )

        elif action_type == "INSERT":
            memory_pattern = re.compile(
                r'(?:MEMORY[_ ]ITEM|NEW[_ ]MEMORY|CONTENT|MEMORY)\s*(?::|=|-)?\s*(.+?)(?=[\s,;]*'
                r'(?:REASONING\s*(?::|=|-)?|MEMORY[_ ]ITEM\s*(?::|=|-)?|UPDATED[_ ]MEMORY\s*(?::|=|-)?|'
                r'MEMORY[_ ]INDEX\s*(?::|=|-)?|ACTION\s*(?::|=|-)?|$))',
                re.IGNORECASE | re.DOTALL
            )
            memory_matches = [m.strip() for m in memory_pattern.findall(block) if m.strip()]
            if not memory_matches:
                fallback_lines = []
                for line in block.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if re.match(r'^(ACTION|MEMORY[_ ]ITEM|UPDATED[_ ]MEMORY|MEMORY[_ ]INDEX|REASONING)\b', stripped, re.IGNORECASE):
                        continue
                    if re.match(r'^\w+\s+block\s*:?\s*$', stripped, re.IGNORECASE):
                        continue
                    stripped = re.sub(r'^[-*•]\s*', '', stripped)
                    stripped = re.sub(r'^\d+\.\s*', '', stripped)
                    if stripped:
                        fallback_lines.append(stripped)
                memory_matches = fallback_lines
            if not memory_matches:
                return ExecutionResult(
                    action_type="NOOP",
                    success=False,
                    reasoning="Failed to parse MEMORY_ITEM for INSERT"
                )

            if len(memory_matches) == 1:
                return ExecutionResult(
                    action_type="INSERT",
                    success=True,
                    memory_content=memory_matches[0],
                    reasoning=reasoning
                )

            results = []
            for mem in memory_matches:
                results.append(ExecutionResult(
                    action_type="INSERT",
                    success=True,
                    memory_content=mem,
                    reasoning=reasoning
                ))
            return results

        elif action_type == "UPDATE":
            index_matches = re.findall(
                r'MEMORY[_ ]INDEX\s*(?::|=|-)?\s*(\d+)',
                block,
                re.IGNORECASE
            )
            update_pattern = re.compile(
                r'UPDATED[_ ]MEMORY\s*(?::|=|-)?\s*(.+?)(?=[\s,;]*(?:REASONING\s*(?::|=|-)?|'
                r'UPDATED[_ ]MEMORY\s*(?::|=|-)?|MEMORY[_ ]ITEM\s*(?::|=|-)?|'
                r'MEMORY[_ ]INDEX\s*(?::|=|-)?|ACTION\s*(?::|=|-)?|$))',
                re.IGNORECASE | re.DOTALL
            )
            update_matches = [m.strip() for m in update_pattern.findall(block) if m.strip()]

            if not index_matches or not update_matches:
                return ExecutionResult(
                    action_type="NOOP",
                    success=False,
                    reasoning="Failed to parse MEMORY_INDEX/UPDATED_MEMORY for UPDATE"
                )

            pair_count = min(len(index_matches), len(update_matches))
            results = []
            for i in range(pair_count):
                memory_index = int(index_matches[i])
                if memory_index >= num_retrieved:
                    results.append(ExecutionResult(
                        action_type="NOOP",
                        success=False,
                        reasoning=f"MEMORY_INDEX {memory_index} out of range [0, {num_retrieved})"
                    ))
                    continue
                results.append(ExecutionResult(
                    action_type="UPDATE",
                    success=True,
                    memory_index=memory_index,
                    memory_content=update_matches[i],
                    reasoning=reasoning
                ))

            if len(results) == 1:
                return results[0]
            return results

        elif action_type == "DELETE":
            index_matches = re.findall(
                r'MEMORY[_ ]INDEX\s*(?::|=|-)?\s*(\d+)',
                block,
                re.IGNORECASE
            )
            if not index_matches:
                return ExecutionResult(
                    action_type="NOOP",
                    success=False,
                    reasoning="Failed to parse MEMORY_INDEX for DELETE"
                )

            results = []
            for idx_str in index_matches:
                memory_index = int(idx_str)
                if memory_index >= num_retrieved:
                    results.append(ExecutionResult(
                        action_type="NOOP",
                        success=False,
                        reasoning=f"MEMORY_INDEX {memory_index} out of range [0, {num_retrieved})"
                    ))
                    continue
                results.append(ExecutionResult(
                    action_type="DELETE",
                    success=True,
                    memory_index=memory_index,
                    reasoning=reasoning
                ))

            if len(results) == 1:
                return results[0]
            return results

        else:
            return ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning=f"Unknown action type: {action_type}"
            )

    def apply_to_memory_bank(self, results: List[ExecutionResult],
                              memory_bank, retrieved_indices: List[int],
                              operation_name: Union[str, List[str], None] = None) -> bool:
        """
        Apply execution results to memory bank.
        Handles multiple results from a single LLM response.

        Note: DELETE operations need special handling because indices shift
        after deletion. We process DELETEs in reverse order of index to avoid
        index invalidation issues.

        Optimization: Batch compute embeddings for all INSERT/UPDATE operations
        at once instead of one by one.

        Args:
            results: List[ExecutionResult]
            memory_bank: MemoryBank object
            retrieved_indices: indices of retrieved memories in memory bank
            operation_name: Operation name or list of names for history tracking
        Returns:
            success: bool (True if all operations succeeded)
        """
        if not results:
            return True

        # Separate operations by type for proper ordering
        inserts = [r for r in results if r.action_type == "INSERT" and r.success]
        updates = [r for r in results if r.action_type == "UPDATE" and r.success]
        deletes = [r for r in results if r.action_type == "DELETE" and r.success]
        # print(len(inserts), len(updates), len(deletes))

        all_success = True

        # Batch compute embeddings for UPDATE and INSERT operations
        # Collect all contents that need embeddings
        update_contents = [r.memory_content for r in updates]
        insert_contents = [r.memory_content for r in inserts]
        all_contents = update_contents + insert_contents

        # Batch compute retriever embeddings
        all_retriever_embeddings = []
        if all_contents:
            try:
                all_retriever_embeddings = get_embeddings(
                    self.retriever_name,
                    all_contents,
                    'context'
                )
            except Exception as e:
                self.logger.warning(f"Failed to batch compute retriever embeddings: {e}")
                all_success = False
                # Fall back to empty embeddings (operations will fail individually)
                all_retriever_embeddings = [None] * len(all_contents)

        # Batch compute state encoder embeddings if encoder is available
        all_state_encoder_embeddings = []
        if all_contents and memory_bank.state_encoder is not None:
            try:
                all_state_encoder_embeddings = memory_bank.state_encoder._encode_texts(all_contents)
            except Exception as e:
                self.logger.warning(f"Failed to batch compute state encoder embeddings: {e}")
                # Fall back to None (memory_bank will compute individually if needed)
                all_state_encoder_embeddings = [None] * len(all_contents)
        else:
            all_state_encoder_embeddings = [None] * len(all_contents)

        # Split embeddings back to update and insert
        update_retriever_embeddings = all_retriever_embeddings[:len(updates)]
        update_state_encoder_embeddings = (
            all_state_encoder_embeddings[:len(updates)] if len(all_state_encoder_embeddings) > 0 else [None] * len(updates)
        )
        insert_retriever_embeddings = all_retriever_embeddings[len(updates):]
        insert_state_encoder_embeddings = (
            all_state_encoder_embeddings[len(updates):] if len(all_state_encoder_embeddings) > 0 else [None] * len(inserts)
        )

        # 1. Process UPDATEs first (before any DELETEs change indices)
        for i, result in enumerate(updates):
            try:
                actual_index = retrieved_indices[result.memory_index]
                retriever_emb = update_retriever_embeddings[i]
                state_encoder_emb = (
                    update_state_encoder_embeddings[i] if i < len(update_state_encoder_embeddings) else None
                )
                if retriever_emb is None:
                    raise ValueError("Retriever embedding is None")
                memory_bank.update_memory(
                    index=actual_index,
                    new_content=result.memory_content,
                    new_embedding=retriever_emb,
                    new_state_encoder_embedding=state_encoder_emb,
                    operation_name=operation_name
                )
            except Exception as e:
                self.logger.warning(f"Failed to apply UPDATE: {e}")
                all_success = False

        # 2. Process DELETEs in reverse order of *actual* memory bank indices to avoid index shift issues
        delete_targets = []
        for result in deletes:
            try:
                actual_index = retrieved_indices[result.memory_index]
                delete_targets.append((actual_index, result))
            except Exception as e:
                self.logger.warning(f"Failed to map DELETE target: {e}")
                all_success = False

        for actual_index, result in sorted(delete_targets, key=lambda x: x[0], reverse=True):
            try:
                memory_bank.delete_memory(index=actual_index)
            except Exception as e:
                self.logger.warning(f"Failed to apply DELETE: {e}")
                all_success = False

        # 3. Process INSERTs last (doesn't affect existing indices)
        for i, result in enumerate(inserts):
            try:
                retriever_emb = insert_retriever_embeddings[i]
                state_encoder_emb = (
                    insert_state_encoder_embeddings[i] if i < len(insert_state_encoder_embeddings) else None
                )
                if retriever_emb is None:
                    raise ValueError("Retriever embedding is None")
                memory_bank.add_memory(
                    content=result.memory_content,
                    embedding=retriever_emb,
                    state_encoder_embedding=state_encoder_emb,
                    metadata={'source': 'inserted'},
                    operation_name=operation_name
                )
            except Exception as e:
                self.logger.warning(f"Failed to apply INSERT: {e}")
                all_success = False

        return all_success
