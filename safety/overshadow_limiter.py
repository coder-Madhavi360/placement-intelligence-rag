from core.interfaces import RetrievalResult


class OvershadowLimiter:
    def __init__(self):
        pass

    def refine(
        self,
        result: RetrievalResult,
        max_tokens: int = 2000
    ) -> RetrievalResult:

        total_tokens = 0
        filtered = []

        for chunk in result.chunks:
            chunk_tokens = len(chunk.text.split())

            if total_tokens + chunk_tokens > max_tokens:
                break

            filtered.append(chunk)
            total_tokens += chunk_tokens

        result.chunks = filtered
        return result

    def limit(self, chunks):
        return chunks