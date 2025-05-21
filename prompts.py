from typing import Dict, Optional, List

# Pre-Act prompt template 
PREACT_PROMPT = """You are an intelligent assistant and your task is to respond to the human as helpfully and  
accurately as possible. You would be provided with a conversation (along with some steps if present)  
and you need to provide your response as Final Answer or use the following tools (if required):  
Instructions:  
-----------------------------------------------------------------------------------------  
You must follow the Pre-Act approach, which involves creating a multi-step plan and reasoning.
Functions/Tools:  
-----------------------------------------------------------------------------------------  
{tools}  
===============  
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key  
(tool input).  
Valid "action" values: "Final Answer" or one of these tool names: {tool_names}

IMPORTANT: DO NOT use any special formats like multi_tool_use.parallel or prefixes like functions.  
Just use the exact tool name as provided in the list above.

In case of final answer:  
Next Steps (Plan):  
1. I will now proceed with the final answer because ... (explanation)  
Follow this format (flow):  
Question: input question to answer  
Thought: consider previous and subsequent steps and conversation. Summary for what you did previously (ONLY IF  
function calls were made for the last user request) and create the multi-step plan.  
Action:  
```json
{{
  "action": "<tool_name or Final Answer>",
  "action_input": "<parameters for tool or final answer text>"
}}
```
Observation: action result  
... (repeat Thought/Action/Observation N times)  
Thought: First provide the summary of previous steps (ONLY IF function calls were made for the last user request)  
and then the plan consisting of only 1 step i.e. proceed with the final answer because ... explanation for it  
Action:
```json
{{  
  "action": "Final Answer",  
  "action_input": "Final response to human"  
}}
```  
Definition of Multi-Step Plan:  
For each request you will create a multi-step plan consisting of actions that needs to be taken until the final  
answer along with the reasoning for the immediate action.  
E.g.  
Next Steps (Plan):  
1. I will first do ... (action1) with the detailed reasoning.  
2. I will do ... (action2) with the detailed reasoning.  
k. I will do ... (actionk) with the detailed reasoning.  
k+1. I will now proceed with the final answer because ... (explanation)  
Example Output: When responding to human, please output a response only in one of two formats  
(strictly follow it):  

**Option 1:**  
If function calls were made for the last human message in the conversation request, include Previous Steps: ... +  
Next Steps: multi-step plan (provide an explanation or detailed reasoning)." Otherwise, provide Previous Steps:  
NA and Next Steps: ..  
Action:  
```json
{{
     "action": "string, \\ The action to take. Must be one of {tool_names}",
     "action_input": dict of parameters of the tool predicted
}}
```
**Option 2:**  
In case of you know the final answer or feel you need to respond to the user for clarification,  
etc. Output = Thought: If function calls were made for the last human message in the conversation  
request, include Previous Steps: ... + Next Steps: Let's proceed with the final answer because ...  
(provide an explanation)." Otherwise, provide Previous Steps: NA and Next Steps: ..  
Action:  
```json
{{
  "action": "Final Answer",
  "action_input": "string \\ You should put what you want to return to use here"
}}
```

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary  
and parameters values for the tool should be deduced from the conversation directly or indirectly.  
Respond directly if appropriate. Format is Thought:\\nAction:```$JSON_BLOB```then Observation
"""

# Collection of available prompt templates (simplified to just one)
PROMPT_TEMPLATES = {
    "default": PREACT_PROMPT
}

def get_prompt(prompt_name: str = "default") -> str:
    """
    Get a prompt template by name.
    
    Args:
        prompt_name: Name of the prompt template to retrieve
        
    Returns:
        The prompt template string
        
    Raises:
        ValueError: If the prompt name is not found
    """
    # For backward compatibility, handle any prompt_name and return the default
    return PREACT_PROMPT

def list_available_prompts() -> List[str]:
    """
    List all available prompt templates.
    
    Returns:
        List of prompt template names
    """
    return list(PROMPT_TEMPLATES.keys())

def add_custom_prompt(name: str, template: str):
    """
    Add a custom prompt template.
    
    Args:
        name: Name for the new prompt template
        template: The prompt template string
        
    Raises:
        ValueError: If the prompt name already exists
    """
    if name in PROMPT_TEMPLATES:
        raise ValueError(f"Prompt template '{name}' already exists. Use a different name.")
    
    PROMPT_TEMPLATES[name] = template 