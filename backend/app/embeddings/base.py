from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for text/image/audio embedding providers."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    async def embed_image(self, image_uri: str) -> list[float]:
        """Override when using a true multimodal embedding model."""
        return await self.embed_text(image_uri)
