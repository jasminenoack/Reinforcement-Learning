from unittest.mock import ANY
from gridworld.components.grid_environment import GridWorldEnv
from gridworld.runner import Runner
from gridworld.utils import RunnerReturn, Step


class FakeEnv(GridWorldEnv): ...


class FakeAgent:
    def __init__(self):
        self.actions = [
            "right",
            "right",
            "right",
            "right",
            "down",
            "down",
            "down",
            "down",
            "down",
        ]
        self._complete = 0
        self._results = []

    def act(self, state):
        return self.actions[len(self._results)]

    def observe(self, step_result):
        self._results.append(step_result)

    def reset(self):
        self._complete = 0
        self._results = []


class TestRunEpisode:
    def test_episode_runs_to_completion(self):
        env = FakeEnv()
        agent = FakeAgent()

        result = Runner(env, agent).run_episode(render=False)

        assert result == RunnerReturn(
            total_reward=93,
            steps=8,
            reached_goal=True,
            trajectory=[
                Step(
                    start=(0, 0),
                    action="right",
                    reward=-1,
                    new_state=(0, 1),
                    done=False,
                ),
                Step(
                    start=(0, 1),
                    action="right",
                    reward=-1,
                    new_state=(0, 2),
                    done=False,
                ),
                Step(
                    start=(0, 2),
                    action="right",
                    reward=-1,
                    new_state=(0, 3),
                    done=False,
                ),
                Step(
                    start=(0, 3),
                    action="right",
                    reward=-1,
                    new_state=(0, 4),
                    done=False,
                ),
                Step(
                    start=(0, 4),
                    action="down",
                    reward=-1,
                    new_state=(1, 4),
                    done=False,
                ),
                Step(
                    start=(1, 4),
                    action="down",
                    reward=-1,
                    new_state=(2, 4),
                    done=False,
                ),
                Step(
                    start=(2, 4),
                    action="down",
                    reward=-1,
                    new_state=(3, 4),
                    done=False,
                ),
                Step(
                    start=(3, 4),
                    action="down",
                    reward=100,
                    new_state=(4, 4),
                    done=True,
                ),
            ],
            visit_counts={
                (0, 0): 1,
                (0, 1): 1,
                (0, 2): 1,
                (0, 3): 1,
                (0, 4): 1,
                (1, 4): 1,
                (2, 4): 1,
                (3, 4): 1,
                (4, 4): 1,
            },
        )


class TestRunEpisodes:
    def test_multiple_episodes_run_to_completion(self):
        env = FakeEnv()
        agent = FakeAgent()

        result = Runner(env, agent).run_episodes(num_episodes=2, render=False)

        assert len(result) == 2
        assert result[0] == RunnerReturn(
            total_reward=93,
            steps=8,
            reached_goal=True,
            trajectory=ANY,
            visit_counts=ANY,
        )
        assert result[1] == RunnerReturn(
            total_reward=93,
            steps=8,
            reached_goal=True,
            trajectory=ANY,
            visit_counts=ANY,
        )


class TestAnalyzeResults:
    def test_analyze_results(self):
        env = FakeEnv()
        agent = FakeAgent()

        results = [
            RunnerReturn(
                total_reward=92,
                steps=8,
                reached_goal=True,
                trajectory=ANY,
            ),
            RunnerReturn(
                total_reward=40,
                steps=42,
                reached_goal=True,
                trajectory=ANY,
            ),
            RunnerReturn(
                total_reward=10,
                steps=98,
                reached_goal=False,
                trajectory=ANY,
            ),
        ]
        analysis = Runner(env, agent).analyze_results(results)
        assert analysis == {
            "reward": {
                "average": 47.333333333333336,
                "max": 92,
                "min": 10,
            },
            "steps": {
                "average": 49.333333333333336,
                "max": 98,
                "min": 8,
            },
            "reached_goal": {
                "count": 2,
            },
        }
