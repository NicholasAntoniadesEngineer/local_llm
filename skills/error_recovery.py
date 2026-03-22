class ErrorRecovery:
    @staticmethod
    def classify_error(error_str: str) -> str:
        error_str = error_str.lower()
        if 'timeout' in error_str or 'timed out' in error_str:
            return 'timeout'
        if 'import' in error_str or 'no module named' in error_str:
            return 'import'
        if 'syntax' in error_str or 'invalid syntax' in error_str:
            return 'syntax'
        if 'network' in error_str or 'connection' in error_str:
            return 'network'
        if 'permission' in error_str or 'access denied' in error_str:
            return 'permission'
        return 'unknown'

    @staticmethod
    def suggest_fix(error_type: str) -> str:
        fixes = {
            'timeout': 'Try increasing timeout duration or checking network connectivity.',
            'import': 'Verify the package is installed and the import path is correct.',
            'syntax': 'Check for syntax errors in the code, such as missing colons or parentheses.',
            'network': 'Check internet connection and try again later.',
            'permission': 'Run with elevated privileges or check file/folder permissions.',
            'unknown': 'Try rephrasing the query or checking system logs for more details.'
        }
        return fixes.get(error_type, 'Unknown error type - try rephrasing the query.')

    @staticmethod
    def should_retry(error_type: str, attempt_num: int) -> bool:
        if error_type == 'syntax':
            return False
        return attempt_num < 3

    @staticmethod
    def backoff_seconds(attempt_num: int) -> float:
        if attempt_num == 0:
            return 1.0
        elif attempt_num == 1:
            return 2.0
        elif attempt_num == 2:
            return 4.0
        return 8.0

if __name__ == "__main__":
    # Test classify_error
    tests = [
        ('TimeoutError', 'timeout'),
        ('ImportError: No module named', 'import'),
        ('SyntaxError: invalid syntax', 'syntax'),
        ('ConnectionError', 'network'),
        ('PermissionError', 'permission'),
        ('UnknownError', 'unknown')
    ]
    
    for error_str, expected in tests:
        result = ErrorRecovery.classify_error(error_str)
        assert result == expected, f"Failed: {error_str} -> {result} (expected {expected})"
        
    # Test should_retry
    assert ErrorRecovery.should_retry('timeout', 0) == True
    assert ErrorRecovery.should_retry('timeout', 1) == True
    assert ErrorRecovery.should_retry('timeout', 2) == True
    assert ErrorRecovery.should_retry('timeout', 3) == False
    assert ErrorRecovery.should_retry('syntax', 0) == False
    
    # Test backoff_seconds
    assert ErrorRecovery.backoff_seconds(0) == 1.0
    assert ErrorRecovery.backoff_seconds(1) == 2.0
    assert ErrorRecovery.backoff_seconds(2) == 4.0
    assert ErrorRecovery.backoff_seconds(3) == 8.0
    
    print('ALL TESTS PASSED')