from smart_router import SmartRouter
from self_evaluator import SelfEvaluator
import json
import os
import difflib
from typing import List, Tuple, Optional

class Orchestrator:
    def __init__(self):
        self.router = SmartRouter()
        self.evaluator = SelfEvaluator()
        self.memory = {}
        self.phase = 'plan'
        self.progress = 0.0

    def plan_cycle(self, knowledge: float, capability: float, progress: float) -> dict:
        """Plan the next cycle based on current state"""
        
        # Validate inputs
        if not (0.0 <= knowledge <= 1.0 and 0.0 <= capability <= 1.0 and 0.0 <= progress <= 1.0):
            raise ValueError("All metrics must be between 0.0 and 1.0")
        
        # Determine next action
        action = self.router.pick_tool(knowledge, capability, progress)
        
        # Format tool prompt
        tool_prompt = self.router.format_tool_prompt(action, {
            'query': 'Optimize AI agent system',
            'path': 'agent_system.py',
            'content': 'Optimized AI agent system code',
            'code': 'print("Optimized AI agent system")',
            'cmd': 'python3 agent_system.py'
        })
        
        # Evaluate progress
        evaluation = self.evaluate_progress(knowledge, capability, progress)
        
        return {
            'action': action,
            'tool_prompt': tool_prompt,
            'phase': self.phase,
            'progress': self.progress,
            'evaluation': evaluation
        }

    def decide_next(self, knowledge: float, capability: float, progress: float) -> str:
        """Decide the next action to take"""
        
        # Validate inputs
        if not (0.0 <= knowledge <= 1.0 and 0.0 <= capability <= 1.0 and 0.0 <= progress <= 1.0):
            raise ValueError("All metrics must be between 0.0 and 1.0")
        
        # Determine next action
        action = self.router.pick_tool(knowledge, capability, progress)
        
        # Check if we should change phase
        if self.router.should_change_phase(knowledge, capability, progress):
            self.phase = 'code' if self.phase == 'research' else 'research'
        
        return action

    def evaluate_progress(self, knowledge: float, capability: float, progress: float) -> dict:
        """Evaluate the progress of the optimization"""
        
        # Validate inputs
        if not (0.0 <= knowledge <= 1.0 and 0.0 <= capability <= 1.0 and 0.0 <= progress <= 1.0):
            raise ValueError("All metrics must be between 0.0 and 1.0")
        
        # Calculate progress score
        progress_score = (knowledge + capability + progress) / 3
        
        # Check for loop detection
        self.router.loop_detector.record("eval", str(knowledge), str(progress))
        loop_detected = self.router.loop_detector.is_stuck()
        
        # Evaluate file quality
        file_evaluation = self.evaluator.evaluate_file('agent_system.py')
        
        return {
            'progress_score': progress_score,
            'loop_detected': loop_detected,
            'file_evaluation': file_evaluation
        }

    def post_mortem(self, knowledge: float, capability: float, progress: float) -> dict:
        """Perform a post-mortem analysis of the optimization process"""
        
        # Validate inputs
        if not (0.0 <= knowledge <= 1.0 and 0.0 <= capability <= 1.0 and 0.0 <= progress <= 1.0):
            raise ValueError("All metrics must be between 0.0 and 1.0")
        
        # Calculate final metrics
        final_metrics = {
            'knowledge': knowledge,
            'capability': capability,
            'progress': progress
        }
        
        # Generate analysis report
        analysis = {
            'phase': self.phase,
            'progress': self.progress,
            'final_metrics': final_metrics
        }
        
        return analysis

if __name__ == "__main__":
    o = Orchestrator()

    # Test 1: Plan returns required keys
    r = o.plan_cycle(0.2, 0.8, 0.5)
    assert 'action' in r, "plan_cycle must return action"
    assert 'evaluation' in r, "plan_cycle must return evaluation"
    assert r['action'] == 'web_search', f"Low knowledge should search, got {r['action']}"

    # Test 2: High metrics should save
    o2 = Orchestrator()
    r2 = o2.plan_cycle(0.9, 0.9, 0.9)
    assert r2['action'] == 'write_file', f"High metrics should write, got {r2['action']}"

    # Test 3: decide_next returns a tool name
    o3 = Orchestrator()
    assert o3.decide_next(0.2, 0.8, 0.5) == 'web_search'
    assert o3.decide_next(0.9, 0.9, 0.9) == 'write_file'

    # Test 4: evaluate_progress returns scores
    o4 = Orchestrator()
    ep = o4.evaluate_progress(0.5, 0.5, 0.5)
    assert 'progress_score' in ep
    assert 0.0 <= ep['progress_score'] <= 1.0

    # Test 5: post_mortem returns analysis
    o5 = Orchestrator()
    pm = o5.post_mortem(0.5, 0.5, 0.5)
    assert 'phase' in pm
    assert 'final_metrics' in pm
    assert pm['final_metrics']['knowledge'] == 0.5

    print('ALL TESTS PASSED')
