"""
Grounded Document Assistant for MGC Aurora Heights.

Instead of building a full vector database for just 3 markdown files (~7KB),
I'm loading all documents directly into the system prompt. This guarantees the
LLM sees the complete context on every query, which is necessary for catching
conflicts like the transfer fee (2% in price list vs 2.5% in FAQ) and correctly
refusing questions outside the docs.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load local environment variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# List of source markdown files to inject into the system prompt
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DOC_FILES = [
    ("01_mgc_aurora_heights_brochure.md", "Project Brochure (March 2025)"),
    ("02_price_list_payment_plan.md",     "Price List & Payment Plan (April 2025)"),
    ("03_booking_policy_faq.md",          "Booking Policy & Sales FAQ (May 2025)"),
]


def load_documents() -> str:
    """Read all markdown docs and combine them into a single context string."""
    sections = []
    for filename, label in DOC_FILES:
        path = os.path.join(DOCS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        sections.append(
            f"--- DOCUMENT: {label} (source: {filename}) ---\n{content}\n"
        )
    return "\n".join(sections)


# Core system prompt with rules for grounding, conflicts, refusal, and calculations
SYSTEM_PROMPT = """You are an internal document assistant for MGC Developments' sales team.
You answer questions about MGC Aurora Heights using ONLY the documents provided below.

STRICT RULES — follow these exactly:

1. GROUNDING: Every claim must come from the documents. After your answer, cite the
   source document(s) by filename. Format: [Source: filename]

2. CONFLICTS: If two documents give DIFFERENT information for the same topic,
   you MUST flag it. Show BOTH values with their sources and say:
   "⚠️ The documents disagree on this — please confirm with management."
   DO NOT silently pick one.

3. REFUSAL: If the answer is NOT in any document, say clearly:
   "This information is not available in the provided documents."
   If the documents explicitly say the information should NOT be given
   (e.g. rental yield projections), relay that policy.
   NEVER invent, estimate, or hallucinate an answer.

4. UNCONFIRMED INFO: If a document says something is unconfirmed, pending, or
   under discussion, say so explicitly. Do not present it as confirmed.

5. CALCULATIONS: When a question requires combining numbers (e.g. base price +
   premiums), show your calculation step by step so the salesperson can verify.

6. TONE: Professional, concise, helpful. You are assisting a salesperson who
   needs accurate answers quickly.

DOCUMENTS:

{documents}
"""


def create_assistant():
    """Initializes the LangChain assistant chain using Gemini 2.5 Flash."""
    documents_text = load_documents()
    system_prompt = SYSTEM_PROMPT.format(documents=documents_text)

    # Always grab the latest key from .env
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
    api_key = os.getenv("GOOGLE_API_KEY", "").strip('"\' ')

    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("Valid GOOGLE_API_KEY not found in .env file.")

    # Using temperature 0 for deterministic answers
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

    def ask(question: str) -> str:
        """Sends the question along with the document system prompt to Gemini."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        response = llm.invoke(messages)
        return response.content

    return ask


# Standalone CLI test for the 5 benchmark questions in the prompt
if __name__ == "__main__":
    print("Loading MGC Document Assistant...")
    ask = create_assistant()

    test_questions = [
        "What's the base price of a 2-bed in Block B?",
        "What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?",
        "What's the transfer fee?",
        "What's the rental yield on a 1-bed?",
        "Who is the anchor tenant?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        answer = ask(q)
        print(f"\n{answer}")
