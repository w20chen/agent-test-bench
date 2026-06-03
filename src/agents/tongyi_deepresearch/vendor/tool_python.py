import re
from typing import Dict, Optional, Union, Any
import json5
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_code
from sandbox_fusion import run_code, RunCodeRequest
from requests.exceptions import Timeout
import os
import random
import time
CHINESE_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')


def has_chinese_chars(data: Any) -> bool:
    text = f'{data}'
    return bool(CHINESE_CHAR_RE.search(text))


# Array of sandbox fusion endpoints
SANDBOX_FUSION_ENDPOINTS = []

# Fallback to single endpoint if environment variable exists
if 'SANDBOX_FUSION_ENDPOINT' in os.environ:
    SANDBOX_FUSION_ENDPOINTS = [
        endpoint.strip()
        for endpoint in os.environ['SANDBOX_FUSION_ENDPOINT'].split(',')
        if endpoint.strip()
    ]


@register_tool('PythonInterpreter', allow_overwrite=True)
class PythonInterpreter(BaseToolWithFileAccess):
    name = "PythonInterpreter"
    description = 'Execute Python code in a sandboxed environment. Use this to run Python code and get the execution results.\n**Make sure to use print() for any output you want to see in the results.**\nFor code parameters, use placeholders first, and then put the code within <code></code> XML tags, such as:\n<tool_call>\n{"purpose": <detailed-purpose-of-this-tool-call>, "name": <tool-name>, "arguments": {"code": ""}}\n<code>\nHere is the code.\n</code>\n</tool_call>\n'

    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Must be provided within <code></code> XML tags. Remember to use print() statements for any output you want to see.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        # self.summary_mapping = SummaryMapping()
    
    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            if has_chinese_chars([self.name_for_human, self.name, self.description, self.parameters]):
                fmt = 'The input for this tool should be a Markdown code block.'

            else:
                fmt = 'Enclose the code within triple backticks (`) at the beginning and end of the code.'
        return fmt

    def observation(self, tool: dict, tool_dict: dict, tool_results, empty_mode: bool=False, readpage: bool=False, max_observation_length: int=None, tokenizer=None):
        print('test')
        assert isinstance(tool_results, str), f"result of python code should be str, instead of {type(tool_results)}. {tool_results}"
        return tool_results
    
    @property
    def function(self) -> dict:  
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }

    def call(self, params: Union[str, dict], files= None, timeout = 50, **kwargs) -> str:
        try:
            if isinstance(params, dict):
                code = params.get('code', '') or params.get('raw', '')
                nested_params = params.get('params')
                if not code and isinstance(nested_params, dict):
                    code = nested_params.get('code', '') or nested_params.get('raw', '')
                if not code:
                    code = extract_code(params)
            else:
                code = params
            if not f"{code}".strip():
                return '[Python Interpreter Error]: Empty code.'
            last_error = None
            max_attempts = 5
            if not SANDBOX_FUSION_ENDPOINTS:
                return '[Python Interpreter Error]: No sandbox fusion endpoints configured.'
            for attempt in range(max_attempts):
                try:
                    # Randomly sample an endpoint for each attempt
                    endpoint = random.choice(SANDBOX_FUSION_ENDPOINTS)
                    print(f"Attempt {attempt + 1}/{max_attempts} using endpoint: {endpoint}")
                    
                    code_result = run_code(RunCodeRequest(code=code, language='python', run_timeout=timeout), max_attempts=1, client_timeout=timeout, endpoint=endpoint)
                    print("[Python] Code Result", code_result)
                    result = []
                    if code_result.run_result.stdout:
                        result.append(f"stdout:\n{code_result.run_result.stdout}")
                    if code_result.run_result.stderr:
                        result.append(f"stderr:\n{code_result.run_result.stderr}")
                    if code_result.run_result.execution_time >= timeout-1:
                        result.append("[PythonInterpreter Error] TimeoutError: Execution timed out.")
                    result = '\n'.join(result)
                    print('SUCCESS RUNNING TOOL')
                    return result if result.strip() else 'Finished execution.'

                except Timeout:
                    last_error = f'[Python Interpreter Error] TimeoutError: Execution timed out on endpoint {endpoint}.'
                    print(f"Timeout on attempt {attempt + 1}: {last_error}")
                    if attempt == max_attempts - 1:
                        return last_error
                    continue
                
                except Exception as e:
                    last_error = f'[Python Interpreter Error]: {str(e)} on endpoint {endpoint}'
                    print(f"Error on attempt {attempt + 1}: {last_error}")
                    if attempt == max_attempts - 1:
                        return last_error
                    continue

            return last_error if last_error else '[Python Interpreter Error]: All attempts failed.'

        except Exception as e:
            return f"[Python Interpreter Error]: {str(e)}"

    def call_specific_endpoint(self, params: Union[str, dict], endpoint: str, timeout: Optional[int] = 30, **kwargs) -> tuple:
        """Test a specific endpoint directly"""
        try:
            if type(params) is str:
                params = json5.loads(params)
            code = params.get('code', '')
            if not code:
                code = params.get('raw', '')
            triple_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
            if triple_match:
                code = triple_match.group(1)
        except Exception:
            code = extract_code(params)

        if not code.strip():
            return False, '[Python Interpreter Error]: Empty code.'

        try:
            start_time = time.time()
            code_result = run_code(RunCodeRequest(code=code, language='python', run_timeout=timeout), 
                                 max_attempts=1, client_timeout=timeout, endpoint=endpoint)
            end_time = time.time()
            
            result = []
            if code_result.run_result.stdout:
                result.append(f"stdout:\n{code_result.run_result.stdout}")
            if code_result.run_result.stderr:
                result.append(f"stderr:\n{code_result.run_result.stderr}")
            
            result = '\n'.join(result)
            execution_time = end_time - start_time
            return True, result if result.strip() else 'Finished execution.', execution_time

        except Timeout:
            return False, '[Python Interpreter Error] TimeoutError: Execution timed out.', None
        except Exception as e:
            return False, f'[Python Interpreter Error]: {str(e)}', None
