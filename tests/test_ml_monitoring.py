import numpy as np

from app.ml_monitoring import assign_variant, population_stability_index


def test_ab_assignment_is_stable() -> None:
    assert assign_variant("attempt-1", candidate_percentage=20) == assign_variant(
        "attempt-1", candidate_percentage=20
    )


def test_population_shift_increases_psi() -> None:
    reference = np.linspace(0, 1, 1000)
    stable = population_stability_index(reference, reference.copy())
    shifted = population_stability_index(reference, reference + 2)

    assert stable == 0
    assert shifted > 0.2
