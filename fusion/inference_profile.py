"""
Sensor-Gated Adaptive Inference (Idea 2 — practical reframe).

Reservoir prediction + router score gate Gemma generation parameters.
High-confidence threat → minimal schema, greedy decode, ~20 tokens.
Low-confidence       → full schema with reasoning, ~80 tokens.

Token-throttling at the API layer captures the spirit of attention-head
sparsification (skip work proportional to signal certainty) without needing
llama.cpp tensor-hook patches.
"""
from dataclasses import dataclass


# Field type/enum specs — llama.cpp grammar enforces these exactly.
_FIELD_SCHEMA: dict = {
    "threat_type":      {"type": "string"},
    "confidence":       {"type": "number"},
    "direction_degrees":{"type": "number"},
    "distance_meters":  {"type": "number"},
    "urgency":          {"type": "string", "enum": ["low", "medium", "high", "critical"]},
    "haptic_pattern":   {"type": "string", "enum": ["none", "slow_pulse", "rapid_left", "rapid_right", "rapid_center"]},
    "natural_language": {"type": "string"},
}


# GBNF fragments per field — strict, no whitespace allowed.
_FIELD_GBNF: dict = {
    "threat_type":      r'"\"threat_type\":" str',
    "confidence":       r'"\"confidence\":" num',
    "direction_degrees":r'"\"direction_degrees\":" num',
    "distance_meters":  r'"\"distance_meters\":" num',
    "urgency":          r'"\"urgency\":" urgency',
    "haptic_pattern":   r'"\"haptic_pattern\":" haptic',
    "natural_language": r'"\"natural_language\":" str',
}


@dataclass(frozen=True)
class InferenceProfile:
    name: str
    max_tokens: int
    temperature: float
    schema_fields: tuple[str, ...]   # fields Gemma must emit (grammar-enforced)

    def json_schema(self) -> dict:
        """Build OpenAI-style json_schema response_format payload.
        llama.cpp converts this to GBNF grammar but allows whitespace → wastes tokens."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"threat_{self.name.lower()}",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {f: _FIELD_SCHEMA[f] for f in self.schema_fields},
                    "required": list(self.schema_fields),
                    "additionalProperties": False,
                },
            },
        }

    def gbnf(self) -> str:
        """Strict no-whitespace GBNF grammar.
        Forces minified JSON: ~40% fewer tokens than json_schema pretty-print."""
        field_rules = ' "," '.join(_FIELD_GBNF[f] for f in self.schema_fields)
        return (
            f'root ::= "{{" {field_rules} "}}"\n'
            r'num ::= "-"? [0-9]+ ("." [0-9]+)?' + "\n"
            r'str ::= "\"" ([^"\\] | "\\" .)* "\""' + "\n"
            r'urgency ::= "\"low\"" | "\"medium\"" | "\"high\"" | "\"critical\""' + "\n"
            r'haptic ::= "\"none\"" | "\"slow_pulse\"" | "\"rapid_left\"" | "\"rapid_right\"" | "\"rapid_center\""'
        )

    def schema_hint(self) -> str:
        """Optional inline hint (kept for backwards compat / token visibility)."""
        return f"FIELDS:{','.join(self.schema_fields)}"


URGENT = InferenceProfile(
    name="URGENT",
    max_tokens=512,   # Gemma 4 thinking model: ~400 reasoning + ~50 JSON output
    temperature=0.0,
    schema_fields=("direction_degrees", "urgency", "haptic_pattern"),
)

STANDARD = InferenceProfile(
    name="STANDARD",
    max_tokens=768,   # more fields, more thinking budget
    temperature=0.05,
    schema_fields=(
        "threat_type", "confidence", "direction_degrees",
        "urgency", "haptic_pattern",
    ),
)

EXPLORATORY = InferenceProfile(
    name="EXPLORATORY",
    max_tokens=80,
    temperature=0.1,
    schema_fields=(
        "threat_type", "confidence", "direction_degrees",
        "distance_meters", "urgency", "haptic_pattern", "natural_language",
    ),
)


def select_profile(router_score: float, reservoir_p: float) -> InferenceProfile:
    """
    Pick profile from highest of router score and reservoir anticipation.
    Router_score: live router._score (0-1).
    Reservoir_p:  predicted threat probability 3 frames ahead (0-1).
    EXPLORATORY removed from demo path — STANDARD is the floor.
    """
    signal = max(router_score, reservoir_p)
    if signal > 0.60:
        return URGENT
    return STANDARD
