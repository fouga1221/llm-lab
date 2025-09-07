"""
Security components: I/O policies, sandboxing, and action validation.
Corresponds to section 8 of the detailed design document.
"""
from typing import List, Tuple, Optional
from urllib.parse import urlparse

from llm_lab_core.core.types import StructuredActions
from llm_lab_core.core.exceptions import SecurityError
from llm_lab_core.tooling.registry import FunctionRegistry
from llm_lab_core.tooling.validation import JsonSchemaValidator
from jsonschema import validate, ValidationError

class IOPolicy:
    """
    Enforces policies about network and file system access.
    """
    def __init__(self, allow_network: bool, allowed_hosts: List[str], allowed_paths: Optional[List[str]] = None) -> None:
        """
        Initializes the I/O policy enforcer.

        Args:
            allow_network: Whether any network access is permitted.
            allowed_hosts: A list of allowed hostnames if network access is on.
            allowed_paths: An optional list of allowed file path prefixes.
        """
        self.allow_network = allow_network
        self.allowed_hosts = [h.lower() for h in allowed_hosts]
        self.allowed_paths = allowed_paths or []

    def validate_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validates a URL against the network policy.

        Args:
            url: The URL to validate.

        Returns:
            A tuple (is_valid, error_message).
        """
        if not self.allow_network:
            return False, "Network access is disabled by policy."
        try:
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname
            if not hostname:
                return False, "URL is malformed or has no hostname."
            if hostname.lower() not in self.allowed_hosts:
                return False, f"Host '{hostname}' is not in the list of allowed hosts."
            return True, None
        except Exception as e:
            return False, f"URL validation failed: {e}"

    def validate_path(self, path: str) -> Tuple[bool, Optional[str]]:
        """
        Validates a file path against the file system policy.

        Args:
            path: The file path to validate.

        Returns:
            A tuple (is_valid, error_message).
        """
        if not self.allowed_paths:
            return False, "File system access is disabled by policy (no allowed paths)."
        
        # Check if the path starts with any of the allowed prefixes
        if not any(path.startswith(prefix) for prefix in self.allowed_paths):
            return False, f"Path '{path}' is not within any of the allowed directories."
            
        # Additional check for path traversal attacks
        if ".." in path:
            return False, "Path traversal attempts ('..') are not allowed."

        return True, None


class Sandbox:
    """
    A placeholder for a sandbox to analyze or dry-run code.
    As per the design, this does NOT execute code.
    """
    def dry_run(self, code: str) -> Tuple[bool, str]:
        """
        Performs a static analysis (dry run) of a code snippet.
        This is a placeholder for a real implementation, which might use
        static analysis tools to detect risky patterns.

        Args:
            code: The code snippet to analyze.

        Returns:
            A tuple (is_safe, message).
        """
        # For now, we'll just check for obviously dangerous keywords.
        dangerous_keywords = ['os.system', 'subprocess', 'eval', 'exec']
        if any(keyword in code for keyword in dangerous_keywords):
            return False, "Code contains potentially dangerous keywords."
        return True, "Code passed basic static analysis."


class ActionValidator:
    """
    A comprehensive validator for StructuredActions, combining schema, I/O,
    and function registry checks.
    """
    def __init__(self, schema_validator: JsonSchemaValidator, io_policy: IOPolicy, registry: FunctionRegistry) -> None:
        """
        Initializes the action validator.

        Args:
            schema_validator: The JSON schema validator.
            io_policy: The I/O policy enforcer.
            registry: The function registry.
        """
        self.schema_validator = schema_validator
        self.io_policy = io_policy
        self.registry = registry

    def validate(self, actions: StructuredActions) -> Tuple[bool, List[str]]:
        """
        Performs a multi-level validation of the structured actions.

        Args:
            actions: The StructuredActions object to validate.

        Returns:
            A tuple (is_valid, list_of_errors).
        """
        all_errors: List[str] = []

        # 1. Basic schema validation
        is_valid, errors = self.schema_validator.validate_actions(actions)
        if not is_valid:
            all_errors.extend(errors)
            return False, all_errors

        # 2. Per-action validation
        for action in actions.get('actions', []):
            func_name = action.get('function')
            if not func_name:
                all_errors.append("Action is missing a 'function' name.")
                continue

            # 2a. Check if function exists in the registry
            spec = self.registry.get(func_name)
            if not spec:
                all_errors.append(f"Function '{func_name}' is not a registered function (hallucination).")
                continue

            # 2b. Validate arguments against the function's specific schema
            args = action.get('arguments', {})
            try:
                validate(instance=args, schema=spec.schema)
            except ValidationError as e:
                all_errors.append(f"Argument validation failed for '{func_name}': {e.message}")
            
            # 2c. Check arguments against I/O policy
            for arg_name, arg_value in args.items():
                if isinstance(arg_value, str):
                    # Heuristic: check if it looks like a URL or path
                    if arg_value.startswith(('http://', 'https://')):
                        is_valid_url, err = self.io_policy.validate_url(arg_value)
                        if not is_valid_url:
                            all_errors.append(f"I/O policy violation for '{func_name}' arg '{arg_name}': {err}")
                    # Heuristic for paths
                    elif '/' in arg_value or '\\' in arg_value:
                        is_valid_path, err = self.io_policy.validate_path(arg_value)
                        if not is_valid_path:
                            all_errors.append(f"I/O policy violation for '{func_name}' arg '{arg_name}': {err}")


        return not all_errors, all_errors
