import difflib
from typing import List, Tuple, Optional
import json
from confidence_scorer import ConfidenceScorer
from loop_detector import LoopDetector
from task_planner import TaskPlanner

class SmartRouter:
    def __init__(self):
        self.confidence_scorer = ConfidenceScorer()
        self.loop_detector = LoopDetector()
        self.task_planner = TaskPlanner("Implement a Smart Tool Router")
        self.current_phase = 'research'
        
    def pick_tool(self, knowledge, capability, progress):
        # Determine the optimal tool based on the current state
        action = self.confidence_scorer.should_act(knowledge, capability, progress)
        
        if action == 'research':
            return 'web_search'
        elif action == 'abort':
            return None
        elif action == 'save':
            return 'write_file'
        else:  # 'code'
            return 'write_file'

    def should_change_phase(self, knowledge, capability, progress):
        # Determine if we should change phases based on the current state
        action = self.confidence_scorer.should_act(knowledge, capability, progress)
        
        if action == 'research':
            return self.current_phase != 'research'
        elif action == 'abort':
            return False
        elif action == 'save':
            return self.current_phase != 'save'
        else:  # 'code'
            return self.current_phase != 'code'

    def format_tool_prompt(self, tool, args):
        # Format the tool prompt based on the tool and arguments
        if tool == 'web_search':
            return f"Perform a web search with query: {args['query']}."
        elif tool == 'write_file':
            return f"Write content to file: {args['path']}."
        elif tool == 'read_file':
            return f"Read content from file: {args['path']}."
        elif tool == 'run_python':
            return f"Run Python code: {args['code']}."
        elif tool == 'bash':
            return f"Execute shell command: {args['cmd']}."
        else:
            return f"Use tool: {tool} with arguments: {args}."

if __name__ == "__main__":
    router = SmartRouter()
    
    # Test 1: Low confidence should trigger search
    assert router.pick_tool(0.2, 0.8, 0.5) == 'web_search', "Test 1 Failed"
    
    # Test 2: High everything should trigger save
    assert router.pick_tool(0.9, 0.9, 0.9) == 'write_file', "Test 2 Failed"
    
    # Test 3: Many errors should trigger abort
    assert router.pick_tool(0.6, 0.2, 0.5) is None, "Test 3 Failed"
    
    # Test 4: Balanced should trigger code (write_file)
    assert router.pick_tool(0.5, 0.5, 0.5) == 'write_file', "Test 4 Failed"
    
    # Test 5: Should change phase from research to code
    assert router.should_change_phase(0.5, 0.5, 0.5) is True, "Test 5 Failed"
    
    # Test 6: Should not change phase if already in code
    router.current_phase = 'code'
    assert router.should_change_phase(0.5, 0.5, 0.5) is False, "Test 6 Failed"
    
    # Test 7: Format tool prompt for web_search
    assert router.format_tool_prompt('web_search', {'query': 'test query'}) == 'Perform a web search with query: test query.', "Test 7 Failed"
    
    # Test 8: Format tool prompt for write_file
    assert router.format_tool_prompt('write_file', {'path': 'test.txt', 'content': 'test content'}) == 'Write content to file: test.txt.', "Test 8 Failed"
    
    print('ALL TESTS PASSED')