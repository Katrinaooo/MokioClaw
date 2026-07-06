import unittest

from game_of_life import GameOfLife


class GameOfLifeRulesTest(unittest.TestCase):
    def test_live_cell_with_fewer_than_two_neighbors_dies(self):
        game = GameOfLife({(0, 0), (0, 1)})

        next_game = game.step()

        self.assertNotIn((0, 0), next_game.live_cells)

    def test_live_cell_with_two_neighbors_survives(self):
        game = GameOfLife({(0, 0), (0, 1), (1, 0)})

        next_game = game.step()

        self.assertIn((0, 0), next_game.live_cells)

    def test_live_cell_with_three_neighbors_survives(self):
        game = GameOfLife({(0, 0), (0, 1), (1, 0), (1, 1)})

        next_game = game.step()

        self.assertIn((0, 0), next_game.live_cells)

    def test_live_cell_with_more_than_three_neighbors_dies(self):
        game = GameOfLife({(0, 0), (0, 1), (1, 0), (1, 1), (-1, 0)})

        next_game = game.step()

        self.assertNotIn((0, 0), next_game.live_cells)

    def test_dead_cell_with_exactly_three_neighbors_becomes_alive(self):
        game = GameOfLife({(0, 1), (1, 0), (1, 1)})

        next_game = game.step()

        self.assertIn((0, 0), next_game.live_cells)

    def test_blinker_oscillates_every_generation(self):
        horizontal = {(-1, 0), (0, 0), (1, 0)}
        vertical = {(0, -1), (0, 0), (0, 1)}
        game = GameOfLife(horizontal)

        self.assertEqual(vertical, game.step().live_cells)
        self.assertEqual(horizontal, game.step().step().live_cells)


if __name__ == "__main__":
    unittest.main()
