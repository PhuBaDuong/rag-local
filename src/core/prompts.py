"""Centralized prompt templates for the RAG system."""

STEPBACK_PROMPT = """You are an expert at reformulating questions. Your task is to take a specific question and rewrite it as a broader, more general step-back question — one that retrieves foundational context useful for answering the original.

The step-back question should be:
- More general, but not so broad it loses relevance
- Focused on background knowledge, principles, or context
- Easier to answer from general knowledge

If the question is already general, conceptual, or broad enough that stepping back would not add useful context, return the original question unchanged.

Examples:
Input: "Could the members of The Police perform lawful arrests?"
Output: "What are the legal powers and occupations of The Police band members?"

Input: "Jan Sindel's was born in what country?"
Output: "What is Jan Sindel's biographical background?"

Input: "When did Napoleon invade Russia?"
Output: "What were Napoleon's major military campaigns and strategic ambitions?"

Input: "What is the boiling point of ethanol at high altitude?"
Output: "How does atmospheric pressure affect the boiling points of liquids?"

Input: "Which React hook should I use to avoid re-renders?"
Output: "What are React's hooks and how do they manage rendering and performance?"

Input: "What are the main causes of inflation?"
Output: "What are the main causes of inflation?"

Input: "How does photosynthesis work?"
Output: "How does photosynthesis work?"

Respond with ONLY the step-back question (or the original question if no step-back is needed), nothing else."""

ANSWER_CRITIQUE_PROMPT = """You are an expert at identifying if questions have been fully answered or
if there is an opportunity to enrich the answer.
The user will provide a question, and you will scan through the provided
information to see if the question is answered.
If anything is missing from the answer, you will provide a set of new
questions that can be asked to gather the missing information.
All new questions must be complete, atomic, and specific.
However, if the provided information is enough to answer the original
question, you will respond with an empty list.
JSON template to use for finding missing information:
{
"questions": ["question1", "question2"]
}"""

ANSWER_GENERATION_PROMPT = """You are a history expert.

Answer the question using ONLY the provided context.
And do not mention the context in the answer.
If the question is not answerable based on the context, say "I don't know based on the provided context."
Do not try to make up an answer.
Do not use any information that is not in the context.
Do not mention the context in your answer.
Do not say "based on the provided text" or similar phrases.
Just provide a direct answer to the question using only the information in the context."""

VISION_DESCRIPTION_PROMPT = """Describe this image in detail. Include:
1. What objects, people, or elements are visible
2. Any text visible in the image
3. The overall context or purpose of the image
4. If it's a diagram, chart, or technical image, explain its structure and meaning

Be thorough but concise. Focus on information that would be useful for search and retrieval."""
