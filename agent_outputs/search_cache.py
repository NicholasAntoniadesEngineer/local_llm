import time
from typing import Optional, Dict

class SearchCache:
    def __init__(self):
        self.cache: Dict[str, (str, float)] = {}
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: str) -> Optional[str]:
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
        self.cache[key] = (value, time.time() + ttl_seconds)

    def cleanup(self):
        current_time = time.time()
        expired_keys = [key for key, (value, timestamp) in self.cache.items() if current_time >= timestamp]
        for key in expired_keys:
            del self.cache[key]

    def stats(self):
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
    
    print('ALL TESTS PASSED')

if __name__ == '__main__':
    test_search_cache()