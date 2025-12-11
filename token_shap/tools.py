"""
Tool classes for AgentSHAP - bundling tool definitions with their executors.

This module provides:
- Tool: A dataclass that bundles tool definition + executor
- create_function_tool: Helper to create custom function tools
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional


@dataclass
class Tool:
    """
    A tool that bundles its API definition with its executor.

    Attributes:
        name: Unique identifier for the tool (e.g., "get_weather")
        definition: The tool definition dict sent to the API
        executor: Callable that executes the tool
        description: Human-readable description of what the tool does

    Example:
        weather = Tool(
            name="get_weather",
            definition={...},
            executor=lambda args: fetch_weather(args["city"])
        )
    """
    name: str
    definition: Dict[str, Any]
    executor: Callable[[Dict[str, Any]], str]
    description: str = ""

    def execute(self, arguments: Dict[str, Any]) -> str:
        """
        Execute this tool with the given arguments.

        Args:
            arguments: Dictionary of arguments from the model's tool call

        Returns:
            String result to send back to the model
        """
        return self.executor(arguments)

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}')"

    @staticmethod
    def create_executor(tools: list) -> Callable[[str, Dict[str, Any]], str]:
        """
        Create an executor callback from a list of tools.

        This is used by AgentSHAP to pass to model.generate_with_tools().

        Args:
            tools: List of Tool objects

        Returns:
            Callable that takes (tool_name, arguments) and returns result string
        """
        tool_map = {t.name: t for t in tools}

        def executor(tool_name: str, arguments: Dict[str, Any]) -> str:
            tool = tool_map.get(tool_name)
            if tool is None:
                return f"Error: Unknown tool '{tool_name}'"
            try:
                return tool.execute(arguments)
            except Exception as e:
                return f"Error executing '{tool_name}': {str(e)}"

        return executor


def create_function_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    executor: Callable[[Dict[str, Any]], str]
) -> Tool:
    """
    Create a custom function tool with its executor bundled together.

    This is the recommended way to create tools for AgentSHAP.
    The executor function will be called when the model invokes this tool.

    Args:
        name: Unique function name (e.g., "get_weather", "query_database")
        description: What the function does (shown to the model)
        parameters: JSON schema for the function parameters
        executor: Function that takes arguments dict and returns string result

    Returns:
        Tool object with bundled definition and executor

    Example:
        weather_tool = create_function_tool(
            name="get_weather",
            description="Get current weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            },
            executor=lambda args: fetch_weather_api(args["city"])
        )
    """
    definition = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    }

    return Tool(
        name=name,
        definition=definition,
        executor=executor,
        description=description
    )


def create_tool_from_function(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> Tool:
    """
    Create a Tool from a Python function.

    Args:
        func: The Python function to wrap as a tool
        name: Override function name (defaults to func.__name__)
        description: Override description (defaults to func.__doc__)
        parameters: JSON schema for parameters (required)

    Returns:
        Tool object wrapping the function

    Example:
        def get_weather(city: str) -> str:
            '''Get current weather for a city'''
            return requests.get(f"https://wttr.in/{city}?format=3").text

        weather_tool = create_tool_from_function(
            get_weather,
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        )
    """
    tool_name = name or func.__name__
    tool_description = description or func.__doc__ or f"Execute {tool_name}"

    if parameters is None:
        raise ValueError(
            f"Parameters schema required for function '{tool_name}'. "
            "Please provide a JSON schema for the function parameters."
        )

    def executor(args: Dict[str, Any]) -> str:
        result = func(**args)
        return str(result) if result is not None else ""

    return create_function_tool(
        name=tool_name,
        description=tool_description,
        parameters=parameters,
        executor=executor
    )
