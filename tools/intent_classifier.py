from sentence_transformers import SentenceTransformer, util

class IntentClassifier:

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.intent_examples = {
            "web_search": [
                "latest placement trends",
                "current stock price",
                "recent news",
                "industry trends"
            ],

            "calculator": [
                "calculate cgpa",
                "package ratio",
                "how many days",
                "compute value"
            ],

            "opinion_guard": [
                "which company is better",
                "should i join amazon",
                "recommend company"
            ],

            "rag": [
                "amazon package",
                "microsoft eligibility",
                "placement statistics"
            ]
        }

        self.embeddings = {
            k: self.model.encode(v, convert_to_tensor=True)
            for k, v in self.intent_examples.items()
        }

    def classify(self, query):

        query_emb = self.model.encode(
            query,
            convert_to_tensor=True
        )

        best_intent = "rag"
        best_score = 0

        for intent, emb in self.embeddings.items():

            score = util.cos_sim(
                query_emb,
                emb
            ).max().item()

            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent