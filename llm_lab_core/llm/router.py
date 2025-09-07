"""
Confidence-based routing logic for LLM calls.
Corresponds to section 5.2 of the detailed design document.
"""
from typing import Dict, Any, Optional, Tuple, List

from llm_lab_core.core.types import Message, GenerationParams, StructuredActions, RouteDecision, empty_actions
from llm_lab_core.llm.runners import BaseRunner
# Forward-referencing for type hints to avoid circular imports
from llm_lab_core.tooling.validation import JsonStreamGuard
from llm_lab_core.tooling.build import SelfChecker


class ConfidenceRouter:
    """
    Routes generation requests to different models based on confidence scores
    and orchestrates the generation process including self-correction loops.
    """
    def __init__(self, primary: BaseRunner, fallback: Optional[BaseRunner], threshold: float = 0.78) -> None:
        """
        Initializes the router.

        Args:
            primary: The main, typically faster or more efficient, LLM runner.
            fallback: An optional, typically more powerful, LLM runner for complex cases.
            threshold: The confidence score below which the fallback is triggered.
        """
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold

    def estimate_confidence(self, text: str, actions: StructuredActions, signals: Dict[str, Any]) -> float:
        """
        Estimates the confidence in the generated output based on various signals.
        This is a placeholder for a more sophisticated confidence model.

        Args:
            text: The generated text reply.
            actions: The generated structured actions.
            signals: A dictionary of signals from the SelfChecker.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        # Placeholder logic: e.g., check for hallucinated function names,
        # low-confidence actions, or self-correction markers.
        confidence = signals.get('self_correction_attempts', 0) * -0.2 + 0.9
        if not actions.get('actions'):
            confidence -= 0.1
        return max(0.0, min(1.0, confidence))

    def route_and_generate(
        self,
        messages: List[Message],
        params: GenerationParams,
        guard: 'JsonStreamGuard',
        self_check: 'SelfChecker'
    ) -> Tuple[str, StructuredActions, RouteDecision]:
        """
        The main orchestration logic. It routes the request, generates a response,
        and potentially triggers a fallback and self-correction loop.

        Args:
            messages: The input messages for the LLM.
            params: The generation parameters.
            guard: The JSON stream guard for validating structured output.
            self_check: The self-checker for analyzing the output.

        Returns:
            A tuple containing the final reply, the structured actions, and the routing decision.
        """
        # This is a simplified placeholder for the complex logic described in the design.
        # A full implementation would involve streaming, incremental validation,
        # and a potential re-routing loop.

        # 1. Initial generation with the primary model
        runner = self.primary
        raw_output = runner.generate(messages, params)

        # 2. Extract actions and check for validity (simplified)
        # In a real scenario, this would be part of the streaming process with JsonStreamGuard
        # For now, we simulate it post-generation.
        actions, error = guard.feed(raw_output)
        if error or not actions:
            actions = empty_actions()  # Ensure actions is not None and typed

        # 3. Perform self-check and estimate confidence
        reply = raw_output # Simplified, real version would separate reply from JSON
        signals = self_check.check(reply, actions, self_check.registry)
        confidence = self.estimate_confidence(reply, actions, signals)

        decision = RouteDecision(
            model=runner.model_info(),
            reason="Primary model selection",
            confidence=confidence
        )

        # 4. Fallback logic
        if self.fallback and confidence < self.threshold:
            # If confidence is low, re-route to the fallback model
            fallback_runner = self.fallback
            decision.reason = f"Fallback triggered due to low confidence ({confidence:.2f} < {self.threshold})"
            
            # A real implementation might add a "fix-up" prompt to the messages
            raw_output = fallback_runner.generate(messages, params)
            actions, error = guard.feed(raw_output)
            if error or not actions:
                actions = empty_actions()
            
            reply = raw_output
            signals = self_check.check(reply, actions, self_check.registry)
            final_confidence = self.estimate_confidence(reply, actions, signals)
            
            decision.model = fallback_runner.model_info()
            decision.confidence = final_confidence

        return reply, actions, decision
