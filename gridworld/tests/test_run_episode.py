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
        pass


class TestRunEpisode:
    def test_episode_runs_to_completion(self):
        env = FakeEnv()
        agent = FakeAgent()

        result = Runner(env, agent).run_episode(render=False)

        assert result == RunnerReturn(
            total_reward=92,
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
                    reward=99,
                    new_state=(4, 4),
                    done=True,
                ),
            ],
        )
