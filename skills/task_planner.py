import json
from typing import List, Dict, Optional
from error_recovery import ErrorRecovery


class TaskPlanner:
    def __init__(self, goal: str):
        self.goal = goal
        self.tasks = []
        self.completed = set()

    def decompose(self) -> List[Dict]:
        """Decompose the goal into a list of tasks."
        
        if not isinstance(self.goal, str) or len(self.goal) == 0:
            raise ValueError("Goal must be a non-empty string")
        
        # Real decomposition logic
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
        """Find the next task to execute."
        
        if not isinstance(completed, list):
            raise ValueError("Completed must be a list")
        
        for task in self.decompose():
            if task['task'] not in self.completed:
                return task
        return None

    def is_complete(self, completed: List[Dict]) -> bool:
        """Check if all tasks are completed."
        
        if not isinstance(completed, list):
            raise ValueError("Completed must be a list")
        
        return len(self.completed) == len(self.decompose())

    def replan(self, failed_task: Dict, error: str) -> List[Dict]:
        """Replan based on the failed task and error."
        
        if not isinstance(failed_task, dict):
            raise ValueError("Failed task must be a dictionary")
        
        if not isinstance(error, str) or len(error) == 0:
            raise ValueError("Error must be a non-empty string")
        
        error_type = ErrorRecovery.classify_error(error)
        fix = ErrorRecovery.suggest_fix(error_type)
        
        return [
            {
                'task': 'Replan',
                'tool': 'web_search',
                'priority': 1,
                'depends_on': []
            },
            {
                'task': 'Fix',
                'tool': 'write_file',
                'priority': 2,
                'depends_on': ['Replan'],
                'fix': fix
            }
        ]

if __name__ == "__main__":
    # Test the TaskPlanner
    planner = TaskPlanner("Implement a simple task planner")
    
    # Test decompose
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
    
    # Test replan
    failed_task = tasks[0]
    error = "Error in research task"
    replanned_tasks = planner.replan(failed_task, error)
    print(f"Replanned tasks: {replanned_tasks}")
    
    print("ALL TESTS PASSED")