import time
import functools
import random
from config.settings import API_RATE_LIMITS

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

class RateLimiter:
    def __init__(self):
        self.provider = "gemini"
        self.limits = API_RATE_LIMITS[self.provider]
        self.requests_made = 0
        self.tokens_used = 0
        self.daily_requests = 0
        self.last_request_time = 0
        self.last_reset_time = time.time()
        self.daily_reset_time = time.time()
    
    def check_rate_limit(self, prompt_tokens=0):
        current_time = time.time()
        
        if current_time - self.daily_reset_time >= 86400:  
            self.daily_requests = 0
            self.daily_reset_time = current_time
        
        if self.daily_requests >= self.limits["daily_limit"]:
            wait_time = 86400 - (current_time - self.daily_reset_time)
            print(f"Daily limit exceeded. Waiting {wait_time/3600:.1f} hours...")
            time.sleep(wait_time)
            self.daily_requests = 0
            self.daily_reset_time = time.time()
        
        if current_time - self.last_reset_time >= 60:
            self.requests_made = 0
            self.tokens_used = 0
            self.last_reset_time = current_time
        
        if (self.requests_made >= self.limits["requests_per_minute"] or
            self.tokens_used + prompt_tokens >= self.limits["tokens_per_minute"]):
            
            wait_time = 60 - (current_time - self.last_reset_time) + 1
            print(f"Rate limit exceeded. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            self.requests_made = 0
            self.tokens_used = 0
            self.last_reset_time = time.time()
        
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.limits["request_delay"]:
            time.sleep(self.limits["request_delay"] - time_since_last)
        
        self.requests_made += 1
        self.tokens_used += prompt_tokens
        self.daily_requests += 1
        self.last_request_time = time.time()

def rate_limiter(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        prompt = args[0] if args else ""
        prompt_tokens = estimate_tokens(prompt)
        
        self.rate_limiter.check_rate_limit(prompt_tokens)
        
        return func(self, *args, **kwargs)
    
    return wrapper