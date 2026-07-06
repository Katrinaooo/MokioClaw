from game_of_life import GameOfLife


GLIDER = {
    (1, 0),
    (2, 1),
    (0, 2),
    (1, 2),
    (2, 2),
}


def render(game, min_x=-1, max_x=6, min_y=-1, max_y=6):
    rows = []
    for y in range(min_y, max_y + 1):
        row = "".join("#" if (x, y) in game.live_cells else "." for x in range(min_x, max_x + 1))
        rows.append(row)
    return "\n".join(rows)


def main():
    game = GameOfLife(GLIDER)

    for generation in range(5):
        print(f"Generation {generation}")
        print(render(game))
        if generation != 4:
            print()
        game = game.step()


if __name__ == "__main__":
    main()
