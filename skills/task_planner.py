from typing import List, Dict, Optional
from error_recovery import ErrorRecovery


class TaskPlanner:
    """Manages task decomposition and execution for an agent."""

    def __init__(self, goal: str):
        """Initialize the TaskPlanner with a goal.

        Args:
            goal: The main objective the agent is trying to achieve.
        """
        self.goal = goal
        self.tasks = []
        self.completed = set()
        self.recovery = ErrorRecovery()

    def decompose(self) -> List[Dict]:
        """Decompose the goal into a list of tasks with dependencies and priorities.

        Returns:
            List of tasks with their properties.
        """
        if not self.goal:
            raise ValueError("Goal cannot be empty")

        # Simple decomposition logic
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
        """Find the next task to execute based on completed tasks.

        Args:
            completed: List of tasks that have been completed.

        Returns:
            The next task to execute, or None if no tasks remain.
        """
        if not isinstance(completed, list):
            raise TypeError("Completed must be a list")

        completed_names = set()
        for c in completed:
            if isinstance(c, dict):
                completed_names.add(c.get('task', ''))
            elif isinstance(c, str):
                completed_names.add(c)

        for task in self.decompose():
            if task['task'] not in completed_names:
                return task
        return None

    def is_complete(self, completed: List[Dict]) -> bool:
        """Check if all tasks are completed."""
        if not isinstance(completed, list):
            raise TypeError("Completed must be a list")

        completed_names = set()
        for c in completed:
            if isinstance(c, dict):
                completed_names.add(c.get('task', ''))
            elif isinstance(c, str):
                completed_names.add(c)

        return all(t['task'] in completed_names for t in self.decompose())


    def replan(self, failed_task: Dict, error: str) -> List[Dict]:
        """Replan tasks based on a failed task and error message.

        Args:
            failed_task: The task that failed.
            error: The error message from the failed task.

        Returns:
            List of replanned tasks.
        """
        if not isinstance(failed_task, dict) or not isinstance(error, str):
            raise TypeError("Failed task must be a dict and error must be a string")

        error_type = self.recovery.classify_error(error)
        fix_suggestion = self.recovery.suggest_fix(error_type)

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
                'fix_suggestion': fix_suggestion
            }
        ]


if __name__ == "__main__":
    # Test the TaskPlanner
    planner = TaskPlanner("Implement a simple task planner")

    # Test decompose
    tasks = planner.decompose()
    assert len(tasks) == 5, "Decompose should return 5 tasks"
    assert all('task' in task for task in tasks), "All tasks should have 'task' key"
    assert all('tool' in task for task in tasks), "All tasks should have 'tool' key"
    assert all('priority' in task for task in tasks), "All tasks should have 'priority' key"
    assert all('depends_on' in task for task in tasks), "All tasks should have 'depends_on' key"

    # Test next_task
    completed = []
    next_task = planner.next_task(completed)
    assert next_task['task'] == 'Research', "Next task should be 'Research'"

    # Test is_complete
    all_completed = [{"task": t["task"]} for t in tasks]
    assert planner.is_complete(all_completed), "All tasks should be marked as complete"

    # Test replan
    failed_task = tasks[0]
    error = "Error in research step"
    replanned_tasks = planner.replan(failed_task, error)
    assert len(replanned_tasks) == 2, "Replan should return 2 tasks"
    assert replanned_tasks[0]['task'] == 'Replan', "Replan task should be first"
    assert replanned_tasks[1]['task'] == 'Fix', "Fix task should be second"

    # Test edge cases
    assert planner.next_task([]) is not None, "Next task should not be None"
    assert planner.is_complete([]) is False, "Is complete should be False"

    print("ALL TESTS PASSED")