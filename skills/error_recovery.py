class ErrorRecovery:
    @staticmethod
    def classify_error(error_str: str) -> str:
        """Classify the error type based on the error string."
        
        if not isinstance(error_str, str):
            raise ValueError("Input must be a string")
        
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
        """Provide a suggested fix for the given error type."
        
        if not isinstance(error_type, str):
            raise ValueError("Input must be a string")
        
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
        """Determine if the operation should be retried based on error type and attempt number."
        
        if not isinstance(error_type, str):
            raise ValueError("Error type must be a string")
        if not isinstance(attempt_num, int) or attempt_num < 0:
            raise ValueError("Attempt number must be a non-negative integer")
        
        if error_type == 'syntax':
            return False
        return attempt_num < 3

    @staticmethod
    def backoff_seconds(attempt_num: int) -> float:
        """Calculate the backoff time in seconds for the given attempt number."
        
        if not isinstance(attempt_num, int) or attempt_num < 0:
            raise ValueError("Attempt number must be a non-negative integer")
        
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
        ('UnknownError', 'unknown'),
        ('', 'unknown'),
        (None, 'unknown'),
        ('SyntaxError: invalid syntax', 'syntax'),
        ('Invalid syntax', 'syntax')
    ]
    
    for error_str, expected in tests:
        try:
            result = ErrorRecovery.classify_error(error_str)
            assert result == expected, f"Failed: {error_str} -> {result} (expected {expected})"
        except Exception as e:
            assert False, f"Error in classify_error: {e}"
    
    # Test should_retry
    assert ErrorRecovery.should_retry('timeout', 0) == True
    assert ErrorRecovery.should_retry('timeout', 1) == True
    assert ErrorRecovery.should_retry('timeout', 2) == True
    assert ErrorRecovery.should_retry('timeout', 3) == False
    assert ErrorRecovery.should_retry('syntax', 0) == False
    assert ErrorRecovery.should_retry('timeout', -1) == False
    assert ErrorRecovery.should_retry('timeout', 3) == False
    assert ErrorRecovery.should_retry('syntax', 1) == False
    assert ErrorRecovery.should_retry('network', 0) == True
    assert ErrorRecovery.should_retry('network', 2) == True
    
    # Test backoff_seconds
    assert ErrorRecovery.backoff_seconds(0) == 1.0
    assert ErrorRecovery.backoff_seconds(1) == 2.0
    assert ErrorRecovery.backoff_seconds(2) == 4.0
    assert ErrorRecovery.backoff_seconds(3) == 8.0
    assert ErrorRecovery.backoff_seconds(-1) == 8.0
    
    print('ALL TESTS PASSED')