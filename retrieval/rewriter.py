"""
retrieval/rewriter.py
Query rewriting - expands a user query into multiple search variants.
Handles temporal, eligibility, hiring, and conflict query types.
"""
import re
import logging

logger = logging.getLogger(__name__)

COMPANY_ALIASES = {
    "tcs": "TCS", "infosys": "Infosys", "amazon": "Amazon",
    "google": "Google", "microsoft": "Microsoft", "flipkart": "Flipkart",
    "wipro": "Wipro", "cognizant": "Cognizant", "capgemini": "Capgemini",
    "ibm": "IBM", "adobe": "Adobe", "oracle": "Oracle", "sap": "SAP",
    "hcl": "HCL", "tech mahindra": "Tech Mahindra", "qualcomm": "Qualcomm",
    "intel": "Intel", "samsung": "Samsung R&D", "deloitte": "Deloitte",
    "accenture": "Accenture",
}


def extract_entities(query: str):
    q = query.lower()
    matches = []
    for alias, canonical in COMPANY_ALIASES.items():
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        for match in re.finditer(pattern, q):
            matches.append((match.start(), canonical))

    entities = []
    for _, canonical in sorted(matches):
        if canonical not in entities:
            entities.append(canonical)
    return entities


class QueryRewriter:
    """
    Expands the user query into recall-oriented variants.
    Rule-based + heuristic (no extra LLM call needed for speed).
    """

    def rewrite(self, query: str) -> list[str]:
        variants = [query]
        q = query.lower()
        entities = extract_entities(query)

        def add(variant: str) -> None:
            variant = " ".join(variant.split())
            if variant and variant not in variants:
                variants.append(variant)

        # Expand company aliases while preserving multi-word entities.
        canonical_query = query
        for alias, canonical in sorted(
            COMPANY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
        ):
            canonical_query = re.sub(
                rf"(?<!\w){re.escape(alias)}(?!\w)",
                canonical,
                canonical_query,
                flags=re.IGNORECASE,
            )
        if canonical_query != query:
            add(canonical_query)
            add(canonical_query.replace("?", ""))

        # Eligibility queries
        if any(w in q for w in ["cgpa", "backlog", "eligibility", "qualify", "eligible"]):
            target = " ".join(entities) if entities else query
            add(f"eligibility criteria {target}")
            add(f"minimum CGPA required maximum backlogs allowed package offered {target}")

        # Package queries
        if any(w in q for w in ["package", "salary", "lpa", "pay", "highest", "lowest"]):
            if entities:
                for entity in entities:
                    add(f"{entity} package LPA offered")
                add(" ".join(f"{entity} package" for entity in entities))
            else:
                add(f"package offered LPA company eligibility {query}")
            add(f"package LPA offered {query}")

        # Temporal queries
        if any(w in q for w in ["trend", "increase", "grew", "growth", "2021", "2022", "2023", "2024"]):
            add(f"package trend year growth {query}")

        # Hiring queries
        if any(w in q for w in ["hire", "analyst", "sde", "intern", "officer", "roles"]):
            add(f"hiring distribution SDE Analyst Officer Intern {query}")

        # Comparison queries
        if len(entities) >= 2:
            add(f"comparison between {' and '.join(entities)}")
            if any(w in q for w in ["package", "salary", "lpa", "pay"]):
                add(f"compare companies salary package LPA {query}")

        # Conflict queries
        if any(w in q for w in ["conflict", "discrepancy", "portal", "official", "different"]):
            add(f"conflicting records official portal {query}")

        # Bond / constraint queries
        if any(w in q for w in ["bond", "bond-free", "no bond"]):
            if entities:
                for entity in entities:
                    add(f"{entity} bond period years bond-free")
                add(f"bond period years bond-free {' '.join(entities)}")
            add(f"bond period years bond-free {query}")

        unique = list(dict.fromkeys(variants))  # preserve order, remove dupes
        logger.debug(f"Rewrote '{query}' -> {len(unique)} variants")
        return unique[:8]  # broad enough for multi-company intent coverage
