from app.bayesian_optimization import bayesian_minimize


def test_bayesian_search_improves_joint_threshold_objective() -> None:
    result = bayesian_minimize(
        lambda values: (values[0] - 0.2) ** 2 + (values[1] + 0.4) ** 2,
        [(-1, 1), (-1, 1)],
        initial_points=6,
        iterations=16,
        seed=7,
    )

    assert result.objective < 0.03
    assert result.evaluations == 22
