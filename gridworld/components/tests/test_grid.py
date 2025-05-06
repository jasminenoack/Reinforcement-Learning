from ..grid_environment import GridWorldEnv


class TestGridWorldEnvInit:
    def test_can_create_env(self):
        env = GridWorldEnv(rows=5, cols=5)
        assert env.rows == 5
        assert env.cols == 5
