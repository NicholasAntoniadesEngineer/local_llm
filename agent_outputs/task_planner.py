import json
from typing import List, Dict, Optional
from memory import SessionMemory
from config import AgentConfig

class TaskPlanner:
    def __init__(self, goal: str):
        self.goal = goal
        self.tasks = []
        self.completed = set()

    def decompose(self) -> List[Dict]:
        # Basic decomposition logic
        # This is a placeholder and should be replaced with actual decomposition logic
        return [
            {
                'task': 'Research',
                'tool': 'web_search',
                'priority': 1,
                'depends_on': []
            },
            {
                'task': 'Analyze',
                'tool': 'read_file',
                'priority': 2,
                'depends_on': ['Research']
            },
            {
                'task': 'Code',
                'tool': 'write_file',
                'priority': 3,
                'depends_on': ['Analyze']
            },
            {
                'task': 'Test',
                'tool': 'run_python',
                'priority': 4,
                'depends_on': ['Code']
            },
            {
                'task': 'Save',
                'tool': 'write_file',
                'priority': 5,
                'depends_on': ['Test']
            }
        ]

    def next_task(self, completed: List[Dict]) -> Optional[Dict]:
        # Find the next task to execute
        for task in self.decompose():
            if task['task'] not in self.completed:
                return task
        return None

    def is_complete(self, completed: List[Dict]) -> bool:
        # Check if all tasks are completed
        return len(self.completed) == len(self.decompose())

    def replan(self, failed_task: Dict, error: str) -> List[Dict]:
        # Replan based on the failed task and error
        # This is a placeholder and should be replaced with actual replan logic
        return [
            {
                'task': 'Replan',
                'tool': 'web_search',
                'priority': 1,
                'depends_on': []
            }
        ]

if __name__ == "__main__":
    # Test the TaskPlanner
    planner = TaskPlanner("Implement a simple task planner")
    tasks = planner.decompose()
    print(f"Decomposed {len(tasks)} tasks:")
    for task in tasks:
        print(f"- {task['task']} (tool: {task['tool']})")
    
    # Test next_task
    completed = []
    next_task = planner.next_task(completed)
    if next_task:
        print(f"Next task: {next_task['task']} (tool: {next_task['tool']})")
    else:
        print("No more tasks to execute.")
    
    # Test is_complete
    planner.completed = tasks
    print(f"Is complete? {planner.is_complete(tasks)}")
    
    print("ALL TESTS PASSED")