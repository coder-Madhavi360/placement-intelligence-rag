from core.interfaces import BaseConflictDetector


class PlacementConflictDetector(BaseConflictDetector):
    def detect(self, chunks):
        return []