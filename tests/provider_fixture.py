"""Local provider fixture used only by automated tests."""

from cayleypy.research.providers import Completion


class FixtureProvider:
    """Deterministic responses for research-loop tests."""

    mode = "test_fixture"

    def complete(self, role: str, system: str, payload: dict) -> Completion:
        problem = payload["problem"]
        n = problem["graph"]["n"]
        family = problem["graph"]["family"]
        if role == "critic":
            data = {"assessment": "Finite computations and replayed words only. No universal proof follows.",
                    "gaps": ["No new Lean proof was supplied.", "QK margins are ranking signals, not proof certificates."],
                    "next_steps": ["Derive and formally check a universal bound respecting every paid rotation."]}
        elif role == "synthesizer":
            data = {"summary": "Test fixture completed; inspect exact search evidence and witnesses.",
                    "next_steps": ["Run with configured LLM roles and a fitted QK artifact for new ideas."],
                    "proof_obligations": ["Universal quantifiers remain open; finite support is not a theorem."]}
        else:
            target = problem.get("target") or ([1, 0] + list(range(n - 1, 1, -1)))
            kind = "diameter" if role == "counterexample_hunter" else "path"
            if role == "theorist" and family == "lrx" and n >= 4:
                kind = "witness"
            if role == "holographer":
                kind = "area_probe"
            data = {"title": f"Fixture {role} probe", "claim": "Test the stated finite instance; leave the general conjecture open.",
                    "rationale": "Replay the reflected construction, enumerate small instances, and challenge candidate distance surrogates.",
                    "experiments": [{"kind": kind, **({"target": target} if kind == "path" else {}),
                                     **({"statistic": "inversions"} if kind == "area_probe" else {})}]}
        return Completion(data, "test-fixture", {})
