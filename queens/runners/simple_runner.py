from time import sleep
from queens.agents.generic_agent import Agent
from queens.components.grid import Grid
from queens.dtos import Observation, Result, RunnerReturn
from queens.utils import build_board_array
from queens.agents.random_agent import RandomAgent


RENDER_DELAY = 0.5


class Runner:
    def __init__(self, env: Grid, agent: Agent):
        self.env = env
        self.agent = agent
        self.agent.reset()
        self.env.reset()

    def run_episode(self, *, render: bool = False):
        self.agent.reset()
        self.env.reset()
        trajectory: list[Result] = []
        if render:
            self.env.render()
            sleep(RENDER_DELAY)

        while not self.env.fully_played:
            state = self.env.get_state()
            action = self.agent.act(observation=Observation(board_state=state))
            self.env.step(*action)
            trajectory.append(Result(action=action))
            if render:
                self.env.render()
                sleep(RENDER_DELAY)

        return RunnerReturn(
            trajectory=trajectory,
            solved=self.env.solved,
            board=self.env.board,
            moves=self.env.moves,
        )

    def run_episodes(
        self,
        num_episodes: int,
        *,
        render: bool = False,
    ) -> list[RunnerReturn]:
        results: list[RunnerReturn] = []
        for _ in range(num_episodes):
            result = self.run_episode(render=render)
            results.append(result)
        return results

    def render_analytics(self, results: list[RunnerReturn]):
        solved = sum(1 for result in results if result.solved)
        failed = sum(1 for result in results if not result.solved)
        avg_moves = sum(result.moves for result in results) / len(results)
        print(f"Total solved: {solved}")
        print(f"Total failed: {failed}")
        print(f"Average moves: {avg_moves:.2f}")


if __name__ == "__main__":
    runner = Runner(env=Grid(build_board_array([])), agent=RandomAgent())
    # runner.run_episode(render=True)

    results = runner.run_episodes(num_episodes=100)
    runner.render_analytics(results)
