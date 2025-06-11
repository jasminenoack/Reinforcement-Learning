import random
from collections import defaultdict

from queens.agents.reinforcement_agents import DecayFailingPaths
from queens.utils import build_board_array

board = build_board_array([])


class TestRebuildScores:
    def test_applies_failure_penalty(self):
        rng = random.Random(1)
        agent = DecayFailingPaths(rng=rng)
        path_one = [(0, 0), (1, 1)]
        path_two = [(2, 2), (3, 3), (4, 4)]
        agent.best_options = defaultdict(list, {5: [path_one], 7: [path_two]})
        agent.failing_paths[tuple(path_one)] = 2
        scores = agent._rebuild_scores()
        expected_one = 5 - (1 + 2) * (1 - len(path_one) / 8)
        expected_two = 7 - (1 + 0) * (1 - len(path_two) / 8)
        assert scores[expected_one] == [path_one]
        assert scores[expected_two] == [path_two]
