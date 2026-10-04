from queue import Queue
from typing import List

from hayate.engine.constants import MAX_BATCH_SIZE
from hayate.engine.request import Request


class Scheduler:
    """Move requests from a waiting queue into the active inference batch."""

    def __init__(self, max_batch_size: int = MAX_BATCH_SIZE):
        self.pool: Queue = Queue()
        self.current_batch: List[Request] = []
        self.request_id = 0
        self.max_batch_size = max_batch_size

    def add(self, request: Request) -> None:
        """Add a prepared request to the waiting queue."""
        self.pool.put(request)

    def tick(self) -> List[Request]:
        """Remove completed requests, then admit requests up to the batch limit."""
        # Keep unfinished requests in the batch so they can continue decoding.
        self.current_batch = [r for r in self.current_batch if not r.is_completed]
        # New requests join at each tick; this supports continuous batching.
        while not self.pool.empty() and len(self.current_batch) < self.max_batch_size:
            self.current_batch.append(self.pool.get())
        return self.current_batch
