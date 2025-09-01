import logging
import json
from pathlib import Path
from typing import Optional
from pydantic import ValidationError
from utils.llm_client import llm_client
from config.agents_io import QueryEnhancerInput, QueryEnhancerOutput
import yaml

logger = logging.getLogger(__name__)

class QueryRephraserAgent:
    def __init__(self):
        config_path = Path(__file__).parent / "query_rephrase_config.yaml"
        try:
            with open(config_path , "r") as f:
                config = yaml.safe_load(f)
            self.system_prompt = config["system_prompt"]
        except Exception as e:
            logger.error(f"[QueryRephraserAgent] Failed to load config: {e}")
            self.system_prompt = "You are a Query Rephraser Agent. (default fallback prompt)"

    async def enhance_query(self, input_data: QueryEnhancerInput) -> QueryEnhancerOutput:
        try:
            prompt = f"""User Intent:
{input_data.core_intent}

Conversation Context:
{input_data.context_notes or 'None'}

Return JSON with rephrased developer task, satisfaction flag, and any suggestions.
"""

            response = await llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=self.system_prompt
            )

            if response:
                cleaned = self._extract_json(response)
                parsed = QueryEnhancerOutput.model_validate_json(cleaned)
                parsed.success = True
                parsed.message = "LLM success"
                return parsed

        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"[QueryRephraserAgent] Validation or JSON error: {e}")
            return QueryEnhancerOutput(
                developer_task=input_data.core_intent.strip(),
                is_satisfied=False,
                suggestions=["LLM returned invalid format."],
                success=False,
                message="Validation or JSON parsing error"
            )
        except Exception as e:
            logger.error(f"[QueryRephraserAgent] Unexpected LLM error: {e}")
            return QueryEnhancerOutput(
                developer_task=input_data.core_intent.strip(),
                is_satisfied=False,
                suggestions=["Unexpected system error."],
                success=False,
                message="Unexpected LLM error"
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
