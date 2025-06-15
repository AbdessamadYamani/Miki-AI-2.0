import logging  
from typing import Dict


def _get_token_usage(response) -> Dict[str, int]:
    """Extracts token usage from the LLM response."""
    tokens = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        tokens["prompt_tokens"] = response.usage_metadata.prompt_token_count
        tokens["candidates_tokens"] = response.usage_metadata.candidates_token_count
        tokens["total_tokens"] = response.usage_metadata.total_token_count
        logging.info(f"LLM Token Usage: Prompt={tokens['prompt_tokens']}, Candidates={tokens['candidates_tokens']}, Total={tokens['total_tokens']}")
    else:
        logging.warning("LLM response did not have usage_metadata.")
    return tokens
