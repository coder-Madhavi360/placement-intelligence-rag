"""
retrieval/rewriter.py
Query rewriting — expands a user query into multiple search variants.
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
    words = query.split()
    entities = []
    for word in words:
        if word.lower() in COMPANY_ALIASES:
            entities.append(
                COMPANY_ALIASES[word.lower()]
            )
    return entities

class QueryRewriter:
    """
    Expands the user query into 2-4 variants for better recall.
    Rule-based + heuristic (no extra LLM call needed for speed).
    """

    def rewrite(self, query: str) -> list[str]:
        variants = [query]
        q = query.lower()

        entities = extract_entities(query)

        if len(entities) >= 2:
            variants.append( " ".join( [f"{e} package"for e in entities] ))

        variants.append(
        f"comparison between {' and '.join(entities)}" )
        # Expand company names
        for alias, canonical in COMPANY_ALIASES.items():
            if alias in q:
                variants.append(query.replace(alias, canonical))
                variants.append(query.replace(alias, canonical).replace("?", ""))

        # Eligibility queries
        if any(w in q for w in ["cgpa", "backlog", "eligibility", "qualify", "eligible"]):
            variants.append(f"minimum CGPA required package backlogs allowed {query}")
            variants.append(f"eligibility criteria {query}")

        # Package queries
        if any(w in q for w in ["package", "salary", "lpa", "pay", "highest"]):
            variants.append(f"package LPA offered {query}")

        # Temporal queries
        if any(w in q for w in ["trend", "increase", "grew", "growth", "2021", "2022", "2023", "2024"]):
            variants.append(f"package trend year growth {query}")

        # Hiring queries
        if any(w in q for w in ["hire", "analyst", "sde", "intern", "officer", "roles"]):
            variants.append(f"hiring distribution SDE Analyst Officer Intern {query}")
        

        # Comparison queries
        if any(w in q for w in ["compare","difference","vs","versus"]):
            entities = extract_entities(query)

        if len(entities) >= 2:
            variants.append(" ".join([f"{e} package"for e in entities]) )
            variants.append( f"comparison between {' and '.join(entities)}")
            variants.append(f"compare companies salary package LPA {query}")

        # Conflict queries
        if any(w in q for w in ["conflict", "discrepancy", "portal", "official", "different"]):
            variants.append(f"conflicting records official portal {query}")

        # Bond / constraint queries
        if any(w in q for w in ["bond", "bond-free", "no bond"]):
            variants.append(f"bond period years bond-free {query}")

        unique = list(dict.fromkeys(variants))  # preserve order, remove dupes
        logger.debug(f"Rewrote '{query}' → {len(unique)} variants")
        return unique[:4]  # cap at 4