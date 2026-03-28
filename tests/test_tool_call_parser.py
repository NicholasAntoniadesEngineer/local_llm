import unittest

from src.runtime.tool_call_parser import extract_tool_calls_from_response


class ToolCallParserTests(unittest.TestCase):
    def test_extracts_native_hermes_tool_call(self):
        response = '<tool_call>{"name":"read_file","arguments":{"path":"src/agent.py"}}</tool_call>'

        tool_calls = extract_tool_calls_from_response(response)

        self.assertEqual(tool_calls, [{"name": "read_file", "arguments": {"path": "src/agent.py"}}])

    def test_extracts_fallback_tool_call(self):
        response = '<tool>list_dir</tool><args>{"path":"."}</args>'

        tool_calls = extract_tool_calls_from_response(response)

        self.assertEqual(tool_calls, [{"name": "list_dir", "arguments": {"path": "."}}])

    def test_extracts_python_fallback_from_markdown_block(self):
        response = "Here is code:\n```python\nprint('ok')\n```"

        tool_calls = extract_tool_calls_from_response(response)

        self.assertEqual(tool_calls, [{"name": "run_python", "arguments": {"code": "print('ok')"}}])


if __name__ == "__main__":
    unittest.main()
