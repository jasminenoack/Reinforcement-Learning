import time
from gridworld.agents.generic_agent import Agent
from gridworld.components.grid_environment import GridWorldEnv
from rich.console import Console

from gridworld.utils import RunnerReturn, Step


console = Console()


class Runner:
    def __init__(self, env: GridWorldEnv, agent: Agent):
        self.env = env
        self.agent = agent

    def run_episode(
        self,
        *,
        render: bool = False,
        clear_render: bool = True,
        sleep: float = 0.5,
    ) -> RunnerReturn:
        trajectory: list[Step] = []
        # make sure the environment is in a clean state
        self.env.reset()
        self.agent.reset()
        done = False
        state = self.env.get_state()

        while not self.env.done:
            if clear_render:
                console.clear()
            if render:
                self.env.render()

            action = self.agent.act(state)
            step_result = self.env.step(action)
            new_state, reward, done = step_result
            step_result = Step(
                start=state,
                action=action,
                reward=reward,
                new_state=new_state,
                done=done,
            )
            self.agent.observe(step_result)
            trajectory.append(
                Step(
                    start=state,
                    action=action,
                    reward=reward,
                    new_state=new_state,
                    done=done,
                )
            )
            state = new_state
            if render:
                time.sleep(sleep)

        return RunnerReturn(
            total_reward=self.env.total_reward,
            steps=len(trajectory),
            reached_goal=self.env.done,
            trajectory=trajectory,
        )
