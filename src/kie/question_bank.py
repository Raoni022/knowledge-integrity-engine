QUESTION_BANK = [
    {
        "id": "rag_broad",
        "level": "broad",
        "topic": "rag",
        "question": "Describe a RAG system you built and the main trade-offs.",
        "scar_keywords": ["chunking", "retrieval", "hallucination", "rerank", "latency"],
    },
    {
        "id": "rag_deep",
        "level": "deep",
        "topic": "rag",
        "question": "What broke first when retrieval quality failed, and how did you debug it?",
        "scar_keywords": ["top-k", "embedding drift", "threshold", "recall", "false positive"],
    },
    {
        "id": "infra_broad",
        "level": "broad",
        "topic": "infra",
        "question": "How do you deploy and monitor ML or AI systems in production?",
        "scar_keywords": ["docker", "rollback", "queue", "observability", "cpu"],
    },
    {
        "id": "infra_deep",
        "level": "deep",
        "topic": "infra",
        "question": "Tell me about a production incident and the exact bottleneck you found.",
        "scar_keywords": ["lock", "timeout", "memory", "container", "throttle"],
    },
    {
        "id": "automation_broad",
        "level": "broad",
        "topic": "automation",
        "question": "How have you used automations to reduce manual work?",
        "scar_keywords": ["webhook", "retry", "rate limit", "payload", "dedup"],
    },
    {
        "id": "automation_deep",
        "level": "deep",
        "topic": "automation",
        "question": "What failure mode was hardest to catch in your workflow engine?",
        "scar_keywords": ["idempotency", "race condition", "queue", "dead letter", "429"],
    },
]

ACTIONS = [
    "ASK_BROAD_RAG",
    "ASK_DEEP_RAG",
    "ASK_BROAD_INFRA",
    "ASK_DEEP_INFRA",
    "ASK_BROAD_AUTOMATION",
    "ASK_DEEP_AUTOMATION",
    "CHECK_PROFILE",
    "CHECK_WEB",
    "PASS",
    "FLAG",
    "ESCALATE",
]
