"""
Function/Tool Registry and Specifications.
Corresponds to section 7.1 of the detailed design document.
"""
from typing import NamedTuple, List, Dict, Any, Optional, Literal

class FunctionSpec(NamedTuple):
    """
    A structured representation of a function (tool) that the LLM can call.
    Using NamedTuple for immutability and clarity.
    """
    name: str
    description: str
    schema: Dict[str, Any]  # JSON Schema for parameters
    risk: Literal['low', 'medium', 'high']

class FunctionRegistry:
    """
    Manages the collection of available functions (tools).
    """
    def __init__(self, specs: List[FunctionSpec]) -> None:
        """
        Initializes the registry with a list of function specifications.

        Args:
            specs: A list of FunctionSpec objects.
        """
        self._functions = {spec.name: spec for spec in specs}

    @classmethod
    def from_json(cls, config: Dict[str, Any]) -> 'FunctionRegistry':
        """
        Factory method to create a FunctionRegistry from a JSON config dictionary.
        
        Args:
            config: The dictionary loaded from a functions.json file.
        
        Returns:
            A new FunctionRegistry instance.
        """
        specs = [
            FunctionSpec(
                name=f.get('name', ''),
                description=f.get('description', ''),
                schema=f.get('parameters', {}),
                risk=f.get('risk', 'medium')
            )
            for f in config.get('functions', [])
        ]
        return cls(specs)

    def get(self, name: str) -> Optional[FunctionSpec]:
        """
        Retrieves a function specification by name.

        Args:
            name: The name of the function.

        Returns:
            The FunctionSpec if found, otherwise None.
        """
        return self._functions.get(name)

    def names(self) -> List[str]:
        """Returns a list of all registered function names."""
        return list(self._functions.keys())

    def all_specs(self) -> List[FunctionSpec]:
        """Returns a list of all function specifications."""
        return list(self._functions.values())
