import time
from typing import Optional, Dict

class SearchCache:
    def __init__(self):
        self.cache: Dict[str, (str, float)] = {}
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: str) -> Optional[str]:
        """Retrieve a value from the cache if it exists and hasn't expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() < timestamp:
                self.hit_count += 1
                return value
            else:
                # Entry has expired, remove it
                del self.cache[key]
        self.miss_count += 1
        return None

    def set(self, key: str, value: str, ttl_seconds: int = 300):
        """Store a value in the cache with a time-to-live (TTL)."""
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Key and value must be strings")
        if not isinstance(ttl_seconds, int) or ttl_seconds < 0:
            raise ValueError("TTL must be a non-negative integer")
        self.cache[key] = (value, time.time() + ttl_seconds)

    def cleanup(self):
        """Remove expired entries from the cache."""
        current_time = time.time()
        expired_keys = [key for key, (value, timestamp) in self.cache.items() if current_time >= timestamp]
        for key in expired_keys:
            del self.cache[key]

    def stats(self):
        """Return statistics about cache hits and misses."""
        return {'hits': self.hit_count, 'misses': self.miss_count}

# Test block
def test_search_cache():
    cache = SearchCache()
    
    # Set 3 values
    cache.set('key1', 'value1')
    cache.set('key2', 'value2')
    cache.set('key3', 'value3')
    
    # Get them (hits)
    assert cache.get('key1') == 'value1'
    assert cache.get('key2') == 'value2'
    assert cache.get('key3') == 'value3'
    
    # Get missing key (miss)
    assert cache.get('key4') is None
    
    # Wait 1 second
    time.sleep(1)
    
    # Set with ttl=0
    cache.set('key5', 'value5', ttl_seconds=0)
    
    # Verify expiry
    assert cache.get('key5') is None
    
    # Test edge cases
    try:
        cache.set(123, 'value')
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for non-string key"
    
    try:
        cache.set('key6', 456)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for non-string value"
    
    try:
        cache.set('key7', 'value7', ttl_seconds=-1)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for negative TTL"
    
    print('ALL TESTS PASSED')

if __name__ == '__main__':
    test_search_cache()