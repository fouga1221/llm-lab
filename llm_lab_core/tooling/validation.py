"""
JSON Schema validation and streaming guards.
Corresponds to section 7.2 of the detailed design document.
"""
import json
from typing import Dict, Any, Tuple, Optional, List, cast
from jsonschema import validate, ValidationError

from llm_lab_core.core.types import StructuredActions

class JsonSchemaValidator:
    """
    Validates Python objects against a given JSON schema.
    """
    def __init__(self, actions_schema: Dict[str, Any]) -> None:
        """
        Initializes the validator with the schema for StructuredActions.

        Args:
            actions_schema: The JSON schema for the root StructuredActions object.
        """
        self.actions_schema = actions_schema

    def validate_actions(self, obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates a dictionary against the StructuredActions schema.

        Args:
            obj: The dictionary to validate.

        Returns:
            A tuple containing a boolean (True if valid) and a list of error messages.
        """
        try:
            validate(instance=obj, schema=self.actions_schema)
            return True, []
        except ValidationError as e:
            return False, [str(e)]
        except Exception as e:
            return False, [f"An unexpected error occurred during validation: {e}"]

class JsonStreamGuard:
    """
    A guard to incrementally parse and validate a JSON object from a stream of text chunks.
    This is crucial for handling structured data from streaming LLM outputs.
    """
    def __init__(self, schema: Dict[str, Any]) -> None:
        """
        Initializes the guard.

        Args:
            schema: The JSON schema to validate against.
        """
        self.schema = schema
        self.reset()

    def feed(self, chunk: str) -> Tuple[Optional[StructuredActions], Optional[str]]:
        """
        Feeds a new chunk of text to the guard. It attempts to parse the
        accumulated buffer as JSON.

        Args:
            chunk: The incoming text chunk from the LLM stream.

        Returns:
            A tuple containing:
            - A valid StructuredActions object if parsing and validation succeed.
            - An error message string if parsing or validation fails.
        """
        self.buffer += chunk
        
        # Attempt to find a complete JSON object in the buffer
        # This is a simplified approach. A robust implementation would handle
        # nested structures and escaped characters more carefully.
        try:
            # Find the start and end of a potential JSON object
            start_brace = self.buffer.find('{')
            end_brace = self.buffer.rfind('}')
            
            if start_brace != -1 and end_brace > start_brace:
                json_str = self.buffer[start_brace : end_brace + 1]
                parsed_obj = json.loads(json_str)
                
                # Once parsed, validate against the schema
                validator = JsonSchemaValidator(self.schema)
                is_valid, errors = validator.validate_actions(parsed_obj)
                
                if is_valid:
                    # Type cast to StructuredActions for type safety
                    return cast(StructuredActions, parsed_obj), None
                else:
                    # This is a validation error, not a parsing error.
                    # The JSON is well-formed but doesn't match the schema.
                    return None, f"Schema validation failed: {'; '.join(errors)}"
            
            # Not a complete JSON object yet
            return None, None

        except json.JSONDecodeError:
            # The buffer does not contain valid JSON yet, which is expected during streaming.
            return None, None
        except Exception as e:
            # An unexpected error occurred
            return None, f"An unexpected error occurred in JsonStreamGuard: {e}"

    def reset(self) -> None:
        """Resets the internal buffer."""
        self.buffer = ""
