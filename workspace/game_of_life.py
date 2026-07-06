from collections import Counter


class GameOfLife:
    def __init__(self, live_cells):
        self.live_cells = set(live_cells)

    def step(self):
        neighbor_counts = Counter()

        for cell in self.live_cells:
            for neighbor in self._neighbors(cell):
                neighbor_counts[neighbor] += 1

        next_live_cells = {
            cell
            for cell, count in neighbor_counts.items()
            if count == 3 or (count == 2 and cell in self.live_cells)
        }

        return GameOfLife(next_live_cells)

    @staticmethod
    def _neighbors(cell):
        x, y = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yield (x + dx, y + dy)


if __name__ == "__main__":
    import time

    #  Gosper glider gun pattern
    gun = {
        (0, 0), (0, 1), (1, 0), (1, 1),
        (10, 0), (10, 1), (10, -1),
        (11, 2), (11, -2), (12, 3), (12, -3),
        (13, 3), (13, -3),
        (14, 0),
        (15, 2), (15, -2),
        (16, 1), (16, 0), (16, -1),
        (17, 0),
        (20, -1), (20, -2), (20, -3),
        (21, -1), (21, -2), (21, -3),
        (22, 0), (22, -4),
        (24, 0), (24, 1), (24, -4), (24, -5),
        (34, -2), (34, -3),
        (35, -2), (35, -3),
    }

    game = GameOfLife(gun)
    for gen in range(200):
        #  Render 20x20 viewport
        xs = [c[0] for c in game.live_cells]
        ys = [c[1] for c in game.live_cells]
        min_x, max_x = min(xs) - 1, max(xs) + 2
        min_y, max_y = min(ys) - 1, max(ys) + 2

        print(f"\033[2J\033[HGeneration {gen}  (cells: {len(game.live_cells)})")
        for y in range(min_y, max_y):
            line = ""
            for x in range(min_x, max_x):
                line += "█" if (x, y) in game.live_cells else "·"
            print(line)

        game = game.step()
        time.sleep(0.1)
