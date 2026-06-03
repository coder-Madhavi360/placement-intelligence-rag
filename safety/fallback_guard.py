from core.interfaces import BaseFallbackGuard


class FallbackGuard(BaseFallbackGuard):
    def is_out_of_corpus(self, query, chunks):
        return False