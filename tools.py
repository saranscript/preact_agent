from typing import Dict, List, Any, Optional, Callable, Type
from langchain_core.tools import BaseTool, tool

class ToolRegistry:
    """
    Registry for managing and loading tools dynamically.
    
    This class allows for registering, retrieving, and dynamically loading tools
    for use with the Pre-Act agent.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register_tool(self, tool_instance: BaseTool, categories: Optional[List[str]] = None):
        """
        Register a tool with the registry.
        
        Args:
            tool_instance: The tool instance to register
            categories: Optional list of categories to associate with the tool
        """
        self._tools[tool_instance.name] = tool_instance
        
        # Register categories if provided
        if categories:
            for category in categories:
                if category not in self._categories:
                    self._categories[category] = []
                self._categories[category].append(tool_instance.name)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: The name of the tool to retrieve
            
        Returns:
            The tool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all registered tools.
        
        Returns:
            List of all tool instances
        """
        return list(self._tools.values())
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """
        Get all tools in a specific category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of tool instances in the category
        """
        if category not in self._categories:
            return []
        
        return [self._tools[name] for name in self._categories[category] if name in self._tools]
    
    def get_tool_names(self) -> List[str]:
        """
        Get all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_categories(self) -> List[str]:
        """
        Get all registered categories.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())


# Default tools for the Pre-Act agent

@tool
def get_weather(location: str) -> str:
    """Call to get the weather from a specific location."""
    # This is a placeholder implementation
    if any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's sunny in San Francisco, temperature is 72°F."
    elif "new york" in location.lower():
        return "It's rainy in New York, temperature is 55°F."
    else:
        return f"The weather in {location} is partly cloudy, temperature is 65°F."

@tool
def get_news_headlines(country: str) -> Dict[str, List[str]]:
    """Get the latest news headlines for a specific country."""
    # This is a placeholder implementation
    if "united states" in country.lower() or "us" in country.lower():
        return {
            "headlines": [
                "Biden announces new climate policy",
                "Tech stocks see significant growth",
                "Major sports team wins championship",
                "New health study reveals benefits of exercise"
            ]
        }
    else:
        return {
            "headlines": [
                f"Latest political developments in {country}",
                f"Economic news from {country}",
                f"Sports updates from {country}",
                f"Cultural events in {country}"
            ]
        }

# Helper function to create a default tool registry
def create_default_registry() -> ToolRegistry:
    """
    Create a registry with the default tools.
    
    Returns:
        A tool registry populated with default tools
    """
    registry = ToolRegistry()
    
    # Register default tools with categories
    registry.register_tool(get_weather, categories=["information", "utilities"])
    registry.register_tool(get_news_headlines, categories=["information", "current_events"])
    
    return registry

# Create a global instance of the registry with default tools
default_registry = create_default_registry() 