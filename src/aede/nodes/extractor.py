"""Evidence Extractor node using Gemini API."""

import json
import os
import re

from aede.state import AEDEState, Fact


def extractor(state: AEDEState) -> AEDEState:
    """
    Extract factual claims from retrieved documents using Gemini API.
    """
    query = state["query"]
    documents = state.get("documents", [])

    if not documents:
        return {**state, "facts": [], "workflow_path": state.get("workflow_path", []) + ["extract"]}

    # Build prompt for evidence extraction
    system_prompt = """You are an evidence extraction assistant. Extract factual claims from documents relevant to the query.

Respond ONLY with valid JSON:
{
  "facts": [
    {"claim": "...", "quote": "...", "chunk_id": 0},
    {"claim": "...", "quote": "...", "chunk_id": 1}
  ]
}

If no relevant facts, respond: {"facts": []}"""

    user_prompt = f"Query: {query}\n\nDocuments:\n"
    for i, doc in enumerate(documents):
        user_prompt += f"\n[Document {i}]:\n{doc[:2000] if len(doc) > 2000 else doc}\n"

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {**state, "error": "GEMINI_API_KEY not set", "facts": [], "workflow_path": state.get("workflow_path", []) + ["extract"]}

    # Use official Google Gemini client
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=f"{system_prompt}\n\n{user_prompt}",
        )

        response_text = response.text
        usage_metadata = response.usage_metadata

    except Exception as e:
        return {**state, "error": f"Extraction API error: {str(e)}", "facts": [], "workflow_path": state.get("workflow_path", []) + ["extract"]}

    # Update token tracking
    current_usage = state.get("token_usage", {})
    if usage_metadata:
        current_usage["extractor_input"] = current_usage.get("extractor_input", 0) + (usage_metadata.prompt_token_count or 0)
        current_usage["extractor_output"] = current_usage.get("extractor_output", 0) + (usage_metadata.candidates_token_count or 0)

    # Parse JSON response
    facts: list[Fact] = []
    try:
        # Try direct parse first
        parsed = json.loads(response_text)
        if "facts" in parsed:
            facts = [Fact(claim=str(f.get("claim", "")), quote=str(f.get("quote", "")), chunk_id=int(f.get("chunk_id", 0))) for f in parsed["facts"]]
    except json.JSONDecodeError:
        # Try markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL) or re.search(r"(\{[^}]*\"facts\"\s*:\s*\[[^\]]*\])", response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "facts" in parsed:
                    facts = [Fact(claim=str(f.get("claim", "")), quote=str(f.get("quote", "")), chunk_id=int(f.get("chunk_id", 0))) for f in parsed["facts"]]
            except json.JSONDecodeError:
                pass

    return {
        **state,
        "facts": facts,
        "token_usage": current_usage,
        "workflow_path": state.get("workflow_path", []) + ["extract"],
        "error": None,
    }