from __future__ import annotations

from math import sqrt
from typing import Iterable

from .search import SearchResult, astar, breadth_first_search


RiverState = tuple[int, int, int, int]
LEFT = 0
RIGHT = 1


def _safe_river_state(state: RiverState) -> bool:
    farmer, fox, chicken, grain = state
    fox_eats_chicken = fox == chicken and farmer != fox
    chicken_eats_grain = chicken == grain and farmer != chicken
    return not fox_eats_chicken and not chicken_eats_grain


def river_successors(state: RiverState) -> Iterable[RiverState]:
    farmer_side = state[0]
    cargo_options = [None, 1, 2, 3]
    for cargo in cargo_options:
        next_state = list(state)
        next_state[0] = RIGHT if farmer_side == LEFT else LEFT
        if cargo is not None:
            if state[cargo] != farmer_side:
                continue
            next_state[cargo] = next_state[0]
        candidate = tuple(next_state)
        if _safe_river_state(candidate):
            yield candidate


def demo_river_crossing() -> SearchResult[RiverState]:
    return breadth_first_search(
        start=(LEFT, LEFT, LEFT, LEFT),
        is_goal=lambda state: all(side == RIGHT for side in state),
        successors=river_successors,
    )


PuzzleState = tuple[int, ...]
GOAL: PuzzleState = (0, 1, 2, 3, 4, 5, 6, 7, 8)


def puzzle_successors(state: PuzzleState) -> Iterable[PuzzleState]:
    blank = state.index(0)
    row, col = divmod(blank, 3)
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nr, nc = row + dr, col + dc
        if not (0 <= nr < 3 and 0 <= nc < 3):
            continue
        swap = nr * 3 + nc
        data = list(state)
        data[blank], data[swap] = data[swap], data[blank]
        yield tuple(data)


def hamming(state: PuzzleState) -> float:
    return float(sum(tile != 0 and tile != GOAL[index] for index, tile in enumerate(state)))


def manhattan(state: PuzzleState) -> float:
    total = 0
    for index, tile in enumerate(state):
        if tile == 0:
            continue
        row, col = divmod(index, 3)
        goal_row, goal_col = divmod(tile, 3)
        total += abs(row - goal_row) + abs(col - goal_col)
    return float(total)


def euclidean(state: PuzzleState) -> float:
    total = 0.0
    for index, tile in enumerate(state):
        if tile == 0:
            continue
        row, col = divmod(index, 3)
        goal_row, goal_col = divmod(tile, 3)
        total += sqrt((row - goal_row) ** 2 + (col - goal_col) ** 2)
    return total


def demo_eight_puzzle() -> SearchResult[PuzzleState]:
    start = (1, 4, 2, 3, 0, 5, 6, 7, 8)
    return astar(
        start=start,
        is_goal=lambda state: state == GOAL,
        successors=puzzle_successors,
        heuristic=manhattan,
    )
