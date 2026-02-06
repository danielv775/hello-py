import asyncio
import json
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

from tasks.paged_attention import PAGED_ATTN_PROMPT


"""
The PagedAttention task was tuned to a ~40% success rate 
by constraining the token budget to 2900. This requires the agent 
to produce a precise, vectorized implementation on the first or second attempt, 
effectively penalizing inefficient or logically flawed 'hallucination' loops.
"""
MAX_TOKENS = 2900


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        namespace = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")
        
        response = await client.messages.create(
            model=model, 
            max_tokens=MAX_TOKENS, 
            tools=tools, 
            messages=messages
        )

        # 1. Check for 'max_tokens' first
        if response.stop_reason == "max_tokens":
            print(
                f"\n[LIMIT REACHED] Model hit the {MAX_TOKENS} token limit. "
                "This iteration will be marked as a Failure."
            )
            # Returning None ensures the 'run_single_test' function records a FAILURE
            return None 

        # 2. Support only valid continuing reasons
        if response.stop_reason not in ["tool_use", "end_turn"]:
            print(f"!!! Warning: Unexpected stop_reason: {response.stop_reason}")
            return None

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    # Extract arguments based on tool
                    handler = tool_handlers[tool_name]
                    tool_input = content.input
                
                # Call the appropriate tool handler
                if tool_name == "python_expression":
                    # Use .get() to avoid crashing if the model makes a formatting error
                    expression = tool_input.get("expression") if isinstance(tool_input, dict) else tool_input
                    
                    if not expression or not isinstance(expression, str):
                        result = {"result": None, "error": "Invalid or missing 'expression' input."}
                    else:
                        try:
                            # Wrap the execution in a try block
                            result = handler(expression) 
                            if verbose:
                                print("\nInput Code:")
                                print("```python")
                                print(expression)
                                print("```")
                                                
                            if verbose:
                                print("\nOutput:")
                                print("```")
                                print(result)
                                print("```")

                        except Exception as e:
                            # Capture the crash and send it back to the model as an observation
                            result = {"result": None, "error": f"Python Execution Error: {str(e)}"}
                            if verbose:
                                print(f"!!! Model's code crashed: {e}")

                elif tool_name == "submit_answer":
                    # Extract answer safely
                    answer = tool_input.get("answer") if isinstance(tool_input, dict) else tool_input
                    
                    if answer is None:
                        result = {"answer": None, "error": "No answer provided to submit_answer."}
                    else:
                        result = handler(answer)
                        submitted_answer = result["answer"]

                else:
                    # Generic handler call
                    try:
                        if isinstance(tool_input, dict):
                            result = handler(**tool_input)
                        else:
                            result = handler(tool_input)
                    except Exception as e:
                        result = {"error": f"Tool execution failed: {str(e)}"}

                # IMPORTANT: This must remain active so the model sees the results!
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": json.dumps(result),
                    }
                )



        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return it
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            # No tool use, conversation might be complete
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    expected_answer: Any,
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=20,
        verbose=verbose,
    )

    success = result == expected_answer

    if success:
        print(f"✓ Run {run_id}: SUCCESS - Got {result}")
    else:
        print(f"✗ Run {run_id}: FAILURE - Got {result}, expected {expected_answer}")

    return run_id, success, result


async def main(concurrent: bool = True):
    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Will be passed to exec(). Use print() to output something. Returns stdout. ",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer to submit"}},
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    # Run the test 10 times and track success rate
    num_runs = 10
    expected_answer = "VERIFIED"
    prompt = PAGED_ATTN_PROMPT

    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 60)

    # Create all test coroutines
    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            expected_answer=expected_answer,
            verbose=False,
        )
        for i in range(num_runs)
    ]

    # Run concurrently or sequentially based on the flag
    if concurrent:
        # Process results as they complete
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        # Run sequentially by awaiting each task in order
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    # Count successes
    successes = sum(success for _, success, _ in results)

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=False))
