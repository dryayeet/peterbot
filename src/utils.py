import time
from collections import deque
from typing import Callable, Any

class RateLimiter:
    """Simple rate limiter using sliding window"""
    def __init__(self, max_calls: int, period: int = 60):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
    
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            now = time.time()
            
            # Remove old calls outside the window
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            # Check if we're at the limit
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Clean up again after sleeping
                    now = time.time()
                    while self.calls and self.calls[0] < now - self.period:
                        self.calls.popleft()
            
            # Record this call
            self.calls.append(time.time())
            return func(*args, **kwargs)
        
        return wrapper

def format_chunk_display(chunk: dict) -> str:
    """Format chunk for display in UI"""
    return f"""**{chunk['book']}** (Page {chunk['page']})
{chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}
*Similarity Score: {chunk.get('similarity_score', 0):.4f}*
"""
