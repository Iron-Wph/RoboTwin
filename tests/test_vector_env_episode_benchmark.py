"""Unit tests for complete-episode benchmark accounting."""

from __future__ import annotations

import unittest

import numpy as np

from test_vector_env_baseline import run_until_done


class FakeVectorEnv:
    def __init__(self) -> None:
        self.calls = 0
        self.actions = []

    def step(self, actions):
        self.calls += 1
        self.actions.append(actions.copy())
        observations = [
            {"state": np.full(14, self.calls + env_id, dtype=np.float32)}
            for env_id in range(2)
        ]
        terminated = [np.array([self.calls >= 2]), np.array([False])]
        truncated = [np.array([False]), np.array([self.calls >= 3])]
        return observations, [0.0, 0.0], terminated, truncated, [{}, {}]


class CompleteEpisodeBenchmarkTest(unittest.TestCase):
    def test_runs_until_all_slots_are_done_and_counts_action_steps(self) -> None:
        env = FakeVectorEnv()
        initial_observations = [
            {"state": np.zeros(14, dtype=np.float32)},
            {"state": np.ones(14, dtype=np.float32)},
        ]

        observations, stats = run_until_done(
            env, initial_observations, horizon=2, max_action_chunks=5
        )

        self.assertEqual(env.calls, 3)
        self.assertEqual(stats["action_chunks"], 3)
        self.assertEqual(stats["completed_envs"], 2)
        self.assertEqual(stats["terminated_envs"], 1)
        self.assertEqual(stats["truncated_envs"], 1)
        self.assertEqual(stats["per_env_action_steps"], [4, 6])
        self.assertEqual(stats["completed_action_steps"], 10)
        self.assertEqual(env.actions[0].shape, (2, 2, 14))
        np.testing.assert_array_equal(env.actions[1][0, 0], np.ones(14))
        np.testing.assert_array_equal(observations[1]["state"], np.full(14, 4))


if __name__ == "__main__":
    unittest.main()
