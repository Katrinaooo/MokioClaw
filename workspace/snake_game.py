"""
简易贪吃蛇游戏（Python 标准库版）

运行图形界面：
    python snake_game.py

运行终端自动演示（无需键盘操作）：
    python snake_game.py --demo

玩法：方向键控制蛇移动；吃到红色食物得分；撞墙或撞到自己游戏结束；结束后按空格重开。
"""

import argparse
import random
import time
from collections import deque

CELL_SIZE = 24
GRID_WIDTH = 24
GRID_HEIGHT = 18
INITIAL_SPEED_MS = 130

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class SnakeCore:
    """游戏核心逻辑，供 GUI 和终端演示共用。"""

    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        cx, cy = self.width // 2, self.height // 2
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.score = 0
        self.game_over = False
        self.food = self._new_food()

    def _new_food(self):
        empty = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.snake
        ]
        return random.choice(empty) if empty else None

    def set_direction(self, direction):
        if direction != OPPOSITE[self.direction]:
            self.next_direction = direction

    def step(self):
        if self.game_over:
            return

        self.direction = self.next_direction
        hx, hy = self.snake[0]
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)

        # 撞墙
        if not (0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height):
            self.game_over = True
            return

        # 如果不是吃食物，尾巴会移动，所以允许新头落到当前尾巴位置
        will_grow = new_head == self.food
        body_to_check = list(self.snake) if will_grow else list(self.snake)[:-1]
        if new_head in body_to_check:
            self.game_over = True
            return

        self.snake.appendleft(new_head)
        if will_grow:
            self.score += 1
            self.food = self._new_food()
        else:
            self.snake.pop()


class SnakeTkApp:
    def __init__(self):
        import tkinter as tk

        self.tk = tk
        self.core = SnakeCore()
        self.root = tk.Tk()
        self.root.title("简易贪吃蛇 - Snake")
        self.root.resizable(False, False)

        canvas_w = GRID_WIDTH * CELL_SIZE
        canvas_h = GRID_HEIGHT * CELL_SIZE
        self.canvas = tk.Canvas(self.root, width=canvas_w, height=canvas_h, bg="#111827")
        self.canvas.pack()

        self.info = tk.Label(
            self.root,
            text="方向键移动 | 空格重开",
            font=("Microsoft YaHei", 12),
            bg="#1f2937",
            fg="white",
            pady=8,
        )
        self.info.pack(fill="x")

        self.root.bind("<Up>", lambda _e: self.core.set_direction(UP))
        self.root.bind("<Down>", lambda _e: self.core.set_direction(DOWN))
        self.root.bind("<Left>", lambda _e: self.core.set_direction(LEFT))
        self.root.bind("<Right>", lambda _e: self.core.set_direction(RIGHT))
        self.root.bind("<space>", self.restart)
        self.draw()
        self.root.after(INITIAL_SPEED_MS, self.tick)

    def restart(self, _event=None):
        self.core.reset()
        self.draw()

    def tick(self):
        if not self.core.game_over:
            self.core.step()
            self.draw()
        self.root.after(INITIAL_SPEED_MS, self.tick)

    def draw_cell(self, x, y, color, outline="#0f172a"):
        pad = 2
        self.canvas.create_rectangle(
            x * CELL_SIZE + pad,
            y * CELL_SIZE + pad,
            (x + 1) * CELL_SIZE - pad,
            (y + 1) * CELL_SIZE - pad,
            fill=color,
            outline=outline,
        )

    def draw(self):
        self.canvas.delete("all")

        # 网格背景
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                self.canvas.create_rectangle(
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    (x + 1) * CELL_SIZE,
                    (y + 1) * CELL_SIZE,
                    fill="#111827",
                    outline="#172033",
                )

        if self.core.food:
            self.draw_cell(*self.core.food, color="#ef4444")

        for i, (x, y) in enumerate(self.core.snake):
            self.draw_cell(x, y, color="#22c55e" if i else "#84cc16")

        status = f"得分：{self.core.score} | 方向键移动 | 空格重开"
        if self.core.game_over:
            status = f"游戏结束！得分：{self.core.score} | 按空格重开"
            self.canvas.create_text(
                GRID_WIDTH * CELL_SIZE // 2,
                GRID_HEIGHT * CELL_SIZE // 2,
                text="GAME OVER",
                fill="white",
                font=("Arial", 32, "bold"),
            )
        self.info.config(text=status)

    def run(self):
        self.root.mainloop()


def choose_demo_direction(core):
    """给终端演示用的简单自动寻路：优先朝食物移动，否则找安全方向。"""
    head = core.snake[0]
    food = core.food
    candidates = []
    if food:
        fx, fy = food
        hx, hy = head
        if fx > hx:
            candidates.append(RIGHT)
        elif fx < hx:
            candidates.append(LEFT)
        if fy > hy:
            candidates.append(DOWN)
        elif fy < hy:
            candidates.append(UP)
    candidates.extend([RIGHT, DOWN, LEFT, UP])

    for d in candidates:
        if d == OPPOSITE[core.direction]:
            continue
        hx, hy = head
        nx, ny = hx + d[0], hy + d[1]
        if 0 <= nx < core.width and 0 <= ny < core.height and (nx, ny) not in list(core.snake)[:-1]:
            return d
    return core.direction


def render_ascii(core):
    chars = [[" ." for _ in range(core.width)] for _ in range(core.height)]
    if core.food:
        fx, fy = core.food
        chars[fy][fx] = " F"
    for i, (x, y) in enumerate(core.snake):
        chars[y][x] = " H" if i == 0 else " S"
    border = "=" * (core.width * 2 + 2)
    rows = [border]
    rows.extend("|" + "".join(row) + "|" for row in chars)
    rows.append(border)
    rows.append(f"Score: {core.score}  Length: {len(core.snake)}")
    rows.append("Legend: H=蛇头, S=蛇身, F=食物")
    return "\n".join(rows)


def run_demo(steps=12, delay=0.15):
    random.seed(7)
    core = SnakeCore(width=16, height=10)
    print("终端自动演示：贪吃蛇会自动朝食物移动。\n")
    for i in range(steps):
        print(f"\n--- Frame {i + 1}/{steps} ---")
        print(render_ascii(core))
        core.set_direction(choose_demo_direction(core))
        core.step()
        if core.game_over:
            print("Game Over!")
            break
        time.sleep(delay)
    print("\n演示结束。想玩图形版请运行：python snake_game.py")


def main():
    parser = argparse.ArgumentParser(description="简易贪吃蛇游戏")
    parser.add_argument("--demo", action="store_true", help="在终端中自动演示若干步")
    parser.add_argument("--steps", type=int, default=12, help="终端演示帧数")
    args = parser.parse_args()

    if args.demo:
        run_demo(steps=args.steps)
    else:
        SnakeTkApp().run()


if __name__ == "__main__":
    main()
