from unittest.mock import PropertyMock

from gridworld.runner import Runner
from gridworld.agents.generic_agent import Agent
from gridworld.components.grid_environment import GridWorldEnv, StepResult, VisitCounter
from gridworld.utils import RunnerReturn, Step


class TestRunEpisodeMock:
    def test_returns_env_values(self, mocker):
        env = mocker.create_autospec(GridWorldEnv, instance=True)
        agent = mocker.create_autospec(Agent, instance=True)

        env.get_state.return_value = (0, 0)
        env.step.return_value = StepResult(new_state=(0, 1), reward=5, done=True)
        env.total_reward = 5
        env.reached_goal = True
        env.visit_counts = VisitCounter({(0, 0): 1})
        type(env).done = PropertyMock(side_effect=[False, True])

        agent.act.return_value = "right"

        runner = Runner(env, agent)
        result = runner.run_episode(render=False)

        assert result == RunnerReturn(
            total_reward=5,
            steps=1,
            reached_goal=True,
            trajectory=[
                Step(
                    start=(0, 0), action="right", reward=5, new_state=(0, 1), done=True
                )
            ],
            visit_counts=VisitCounter({(0, 0): 1}),
        )
        env.reset.assert_called_once_with()
        agent.reset.assert_called_once_with()
        agent.observe.assert_called_once_with(result["trajectory"][0])


class TestRunEpisodesAndAnalyze:
    def test_runs_multiple_and_analyzes(self, mocker):
        env = mocker.create_autospec(GridWorldEnv, instance=True)
        agent = mocker.create_autospec(Agent, instance=True)
        runner = Runner(env, agent)

        results = [
            RunnerReturn(
                total_reward=1,
                steps=2,
                reached_goal=True,
                trajectory=[],
                visit_counts=VisitCounter(),
            ),
            RunnerReturn(
                total_reward=3,
                steps=4,
                reached_goal=False,
                trajectory=[],
                visit_counts=VisitCounter(),
            ),
        ]

        run_episode = mocker.patch.object(
            Runner, "run_episode", autospec=True, side_effect=results
        )

        returned = runner.run_episodes(num_episodes=2, render=False)

        assert returned == results
        assert run_episode.call_count == 2

        analysis = runner.analyze_results(returned)
        assert analysis == {
            "reward": {"average": 2.0, "max": 3, "min": 1},
            "steps": {"average": 3.0, "max": 4, "min": 2},
            "reached_goal": {"count": 1},
        }
