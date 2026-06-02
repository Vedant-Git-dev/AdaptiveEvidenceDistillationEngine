"""Final Reasoner node using Gemini API."""

import os

from aede.state import AEDEState


def final_reasoner(state: AEDEState) -> AEDEState:
    """
    Generate final answer using compressed evidence with Gemini API.
    Targets 300-800 tokens for concise, evidence-based answers.
    """
    query = state["query"]
    compressed_evidence = state.get("compressed_evidence", [])

    # If no compressed evidence, try raw facts as fallback
    if not compressed_evidence:
        facts = state.get("facts", [])
        if facts:
            compressed_evidence = [f"{f['claim']} - \"{f['quote']}\"" if f["quote"] else f["claim"] for f in facts[:20]]
        else:
            return {**state, "answer": "Insufficient evidence to answer the query.", "workflow_path": state.get("workflow_path", []) + ["reason"]}

    evidence_text = "\n".join([f"- {e}" for e in compressed_evidence])

    system_prompt = """You are a helpful assistant answering questions based on evidence.

Guidelines:
- Answer concisely (target 300-800 words)
- Use the evidence provided to support your answer
- If evidence is insufficient, say so
- Cite the sources when possible"""

    user_prompt = f"Query: {query}\n\nEvidence:\n{evidence_text}\n\nProvide a clear, evidence-based answer."

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {**state, "answer": _simple_answer(query, compressed_evidence), "workflow_path": state.get("workflow_path", []) + ["reason"]}

    # Use official Google Gemini client
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"{system_prompt}\n\n{user_prompt}",
        )

        answer = response.text
        usage_metadata = response.usage_metadata

    except Exception as e:
        print(f"Reasoner API error: {str(e)}")
        return {**state, "answer": _simple_answer(query, compressed_evidence), "error": str(e), "workflow_path": state.get("workflow_path", []) + ["reason"]}

    # Update token tracking
    current_usage = state.get("token_usage", {})
    if usage_metadata:
        current_usage["reasoner_input"] = current_usage.get("reasoner_input", 0) + (usage_metadata.prompt_token_count or 0)
        current_usage["reasoner_output"] = current_usage.get("reasoner_output", 0) + (usage_metadata.candidates_token_count or 0)
        current_usage["total"] = usage_metadata.total_token_count

    return {
        **state,
        "answer": answer,
        "token_usage": current_usage,
        "workflow_path": state.get("workflow_path", []) + ["reason"],
        "error": None,
    }


def _simple_answer(query: str, evidence: list[str]) -> str:
    """Generate a simple answer when LLM is unavailable."""
    if not evidence:
        return "No evidence available to answer the query."

    evidence_text = "\n\n".join([f"- {e}" for e in evidence[:10]])

    return f"""Based on the available evidence, here is an answer to: {query}

{evidence_text}

Note: This is an automated summary. For a more refined answer, configure the Gemini API key."""