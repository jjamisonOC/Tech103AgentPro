from openai import OpenAI
from typing import List, Dict, Any
import json
import os
import traceback

from .tools.base import Tool

REACT_AGENT_SYSTEM_PROMPT = """
You are an intelligent AI agent equipped with external tools that you can call to solve problems and answer questions accurately.

You have access to the following tools:

{tools}

IMPORTANT INSTRUCTIONS:

1. Think step-by-step before using any tool.
2. Use tools when:
   - You need real-time or factual information
   - A task requires code execution, file generation, or structured output
   - The answer depends on specialized functionality that a tool provides
3. Choose the most relevant tool from: [{tool_names}]
4. Format your Action Input exactly as required by the tool description.
   - If the input requires JSON or structured text, format it correctly.
   - Do not guess—use valid syntax.
5. After a tool returns a result (Observation), reflect on how it helps answer the question.
6. Use the exact tool output in your Final Answer when appropriate.
7. If no tool provides helpful output, say so in your Final Answer.
8. Never hallucinate or fabricate facts. Always rely on tool output for factual claims.
9. You may use multiple Thought/Action/Observation steps if needed.
10. Always conclude with `Final Answer:`.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer based on the tool results
Final Answer: the final answer to the original input question, incorporating the exact information from the tools

Begin!
"""

class AgentPro:
    def __init__(self, llm = None, tools: List[Tool] = [], system_prompt: str = None, react_prompt: str = REACT_AGENT_SYSTEM_PROMPT):
        """
        Initialize the AgentPro instance.
        
        Args:
            llm: The language model client (defaults to OpenAI)
            tools: List of Tool instances to be used by the agent
            system_prompt: Optional custom system prompt
            react_prompt: ReAct framework prompt template
        """
        super().__init__()
        # Initialize the LLM client
        self.client = llm if llm else OpenAI()
        
        # Debug: Print each tool name before normalization
        print("Tools being registered:")
        for tool in tools:
            print(f"- Original name: '{tool.name}', Normalized: '{tool.name.lower().replace(' ', '_')}'")
        
        # Format and register the tools
        self.tools = self.format_tools(tools)
        
        # Debug: Print registered tool names after formatting
        print("Registered tools:")
        for key, tool in self.tools.items():
            print(f"- Key: '{key}', Tool name: '{tool.name}'")
        
        # Format the react prompt with tool descriptions and names
        self.react_prompt = react_prompt.format(
            tools="\n\n".join(map(lambda tool: tool.get_tool_description(), tools)),
            tool_names=", ".join(map(lambda tool: tool.name, tools))
        )
        
        # Initialize conversation history
        self.messages = []
        
        # Add system prompts to the conversation
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self.messages.append({"role": "system", "content": self.react_prompt})

    def format_tools(self, tools: List[Tool]) -> Dict:
        """
        Format the list of tools into a dictionary mapping tool names to tool instances.
        
        Args:
            tools: List of Tool instances
            
        Returns:
            Dictionary mapping normalized tool names to Tool instances
        """
        tool_names = list(map(lambda tool: tool.name, tools))
        return dict(zip(tool_names, tools))

    def parse_action_string(self, text: str) -> tuple:
        """
        Parses action and action input from a string containing thoughts and actions.
        Handles multi-line actions and optional observations.
        
        Args:
            text: The text to parse for Action and Action Input
            
        Returns:
            Tuple of (action, action_input)
        """
        lines = text.split('\n')
        action = None
        action_input = []
        is_action_input = False

        for line in lines:
            if line.startswith('Action:'):
                action = line.replace('Action:', '').strip()
                continue

            if line.startswith('Action Input:'):
                is_action_input = True
                # Handle single-line action input
                input_text = line.replace('Action Input:', '').strip()
                if input_text:
                    action_input.append(input_text)
                continue

            if line.startswith('Observation:'):
                is_action_input = False
                continue

            # Collect multi-line action input
            if is_action_input and line.strip():
                action_input.append(line.strip())

        # Join multi-line action input
        action_input = '\n'.join(action_input)
        
        # Try to parse as JSON if possible
        try:
            action_input = json.loads(action_input)
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON, keep as string
            pass
            
        return action, action_input

    def tool_call(self, response: str) -> str:
        """
        Process a tool call from the agent's response.
        
        Args:
            response: The agent's response containing Action and Action Input
            
        Returns:
            Observation string with the tool's output
        """
        action, action_input = self.parse_action_string(response)
        print(f"\nDEBUG - Tool Call: '{action}'")
        print(f"DEBUG - Tool Input: '{action_input}'")
        
        try:
            if action and action.strip().lower() in self.tools:
                tool = self.tools[action.strip().lower()]
                print(f"DEBUG - Using tool: {tool.name}")
                
                # Execute the tool and capture the result
                print(f"DEBUG - Executing tool...")
                tool_observation = tool.run(action_input)
                
                # Print the actual response from the tool
                print(f"\nDEBUG - Tool observation:")
                print("-" * 50)
                print(tool_observation[:500] + "..." if len(tool_observation) > 500 else tool_observation)
                print("-" * 50)
                
                return f"Observation: {tool_observation}"
            else:
                available_tools = list(self.tools.keys())
                print(f"DEBUG - Tool not found. Action: '{action}', Available tools: {available_tools}")
                return f"Observation: Tool '{action}' not found. Available tools: {available_tools}"
        except Exception as e:
            print(f"DEBUG - Tool execution error: {str(e)}")
            print(traceback.format_exc())
            return f"Observation: There was an error executing the tool\nError: {e}"

    def __call__(self, prompt: str) -> str:
        """
        Run the agent on a user prompt.
        
        Args:
            prompt: The user's input prompt
            
        Returns:
            The agent's final answer
        """
        # Add the user prompt to the conversation
        self.messages.append(
            {"role": "user", "content": prompt}
        )
        
        # Initialize variables for the conversation loop
        response = ""
        step_count = 0
        max_steps = 10  # Safety limit to prevent infinite loops
        
        print(f"\nAvailable tools: {list(self.tools.keys())}")
        print(f"User prompt: {prompt}")
        
        while step_count < max_steps:
            step_count += 1
            print(f"\n--- Step {step_count} ---")
            
            # Get assistant response
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4.1",
                    messages=self.messages,
                    max_tokens=2000
                ).choices[0].message.content.strip()
            except Exception as e:
                print(f"Error getting model response: {e}")
                return f"Error: Could not get a response from the model - {e}"
                
            self.messages.append({"role":"assistant", "content": response})
            
            # Print the response
            print("\nAssistant response:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # Check if we've reached a final answer
            if "Final Answer" in response:
                print("\n--- Conversation Complete ---")
                final_answer = response.split("Final Answer:")[-1].strip()
                return final_answer
            
            # Check if there's a tool call
            if "Action" in response and "Action Input" in response:
                observation = self.tool_call(response)
                
                # Add the observation to the messages
                self.messages.append(
                    {"role": "assistant", "content": observation}
                )
                
                print("\nAdded observation to conversation")
            else:
                print("\nNo tool call detected, but no final answer either. Continuing conversation.")
        
        # If we've reached the maximum steps without a final answer
        print(f"\n--- Reached maximum steps ({max_steps}) without final answer ---")
        return "The agent was unable to provide a conclusive answer after multiple steps."