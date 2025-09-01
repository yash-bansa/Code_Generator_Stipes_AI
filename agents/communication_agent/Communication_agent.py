import logging
from typing import Optional
import json
import yaml
from pathlib import Path
from pydantic import ValidationError
from utils.llm_client import llm_client
from config.agents_io import CommunicationInput, CommunicationOutput

logger = logging.getLogger(__name__)

class CommunicationAgent:
    def __init__(self):
        config_path = Path(__file__).parent / "communication_config.yaml"
        try:
            with open(config_path , "r") as f:
                config = yaml.safe_load(f)
            self.system_prompt = config["system_prompt"]
        except Exception as e:
            logger.error(f"[CommunicationAgent] Failed to load system prompt: {e}")
            self.system_prompt = "You are a communication Agent. (default fallback prompt)"

    async def extract_intent(self, input_data: CommunicationInput) -> CommunicationOutput:
        try:
            full_context = "\n".join(input_data.conversation_history + [input_data.user_query])

            prompt = f"""Conversation History:
{full_context}

Extract:
1. core_intent: (1-line clear dev goal)
2. context_notes: relevant hints from earlier turns
Return as JSON object.
"""

            response = await llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=self.system_prompt
            )

            if response:
                cleaned = self._extract_json(response)
                parsed = CommunicationOutput.model_validate_json(cleaned)
                return parsed

        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"[CommunicationAgent] Validation/Parse error: {e}")
        except Exception as e:
            logger.error(f"[CommunicationAgent] LLM error: {e}")

        return CommunicationOutput(
            core_intent=input_data.user_query.strip(),
            context_notes="",
            success=False,
            message="LLM failed to extract intent"
        )

    def _extract_json(self, response: str) -> str:
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            return response[start:end].strip()
        return response.strip()
