from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query and return results

        Args:
            query: User query
            context: Additional context from other agents

        Returns:
            Dictionary with agent results
        """
        pass

    def get_info(self) -> str:
        """Return agent description"""
        return f"{self.name} Agent"