"""
Configuration loading utilities.
"""
import yaml
import json
from typing import Any, Dict

class ConfigLoader:
    """
    Loads configuration files (YAML, JSON) from specified paths.
    Corresponds to section 4.1 of the detailed design document.
    """
    def __init__(self, model_path: str, security_path: str, functions_path: str) -> None:
        """
        Initializes the loader with paths to the configuration files.

        Args:
            model_path: Path to the model configuration YAML file.
            security_path: Path to the security policy YAML file.
            functions_path: Path to the function definitions JSON file.
        """
        self._model_path = model_path
        self._security_path = security_path
        self._functions_path = functions_path

    def load_model(self) -> Dict[str, Any]:
        """
        Loads and parses the model configuration YAML file.

        Returns:
            A dictionary containing the model configuration.
        """
        with open(self._model_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_security(self) -> Dict[str, Any]:
        """
        Loads and parses the security policy YAML file.

        Returns:
            A dictionary containing the security policy.
        """
        with open(self._security_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_functions(self) -> Dict[str, Any]:
        """
        Loads and parses the function definitions JSON file.

        Returns:
            A dictionary containing the function definitions.
        """
        with open(self._functions_path, 'r', encoding='utf-8') as f:
            return json.load(f)
