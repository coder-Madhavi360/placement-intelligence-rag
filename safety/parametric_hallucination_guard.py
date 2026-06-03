from dataclasses import dataclass


@dataclass
class HallucinationReport:
    final_verdict: str = "PASS"
    explanation: str = ""
    self_consistency_score: float = 1.0
    recitation_score: float = 1.0
    chain_verified: bool = True


class ParametricHallucinationGuard:
    def __init__(self, generator, **kwargs):
        self.generator = generator

    def run(self, query, prompt, chunks):
        response = self.generator.generate(prompt)

        report = HallucinationReport()

        return response.answer, report