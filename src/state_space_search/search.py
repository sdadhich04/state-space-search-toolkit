from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Callable, Generic, Hashable, Iterable, TypeVar


StateT = TypeVar("StateT", bound=Hashable)


@dataclass(frozen=True)
class SearchResult(Generic[StateT]):
    path: list[StateT]
    cost: float
    expanded: int
    max_frontier: int


def _reconstruct_path(
    parent: dict[StateT, StateT | None],
    goal: StateT,
) -> list[StateT]:
    path: list[StateT] = []
    current: StateT | None = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def breadth_first_search(
    start: StateT,
    is_goal: Callable[[StateT], bool],
    successors: Callable[[StateT], Iterable[StateT]],
) -> SearchResult[StateT]:
    """Run graph-search BFS and return a path if a goal is reachable."""
    frontier: deque[StateT] = deque([start])
    parent: dict[StateT, StateT | None] = {start: None}
    seen: set[StateT] = {start}
    expanded = 0
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        state = frontier.popleft()
        if is_goal(state):
            path = _reconstruct_path(parent, state)
            return SearchResult(path, float(len(path) - 1), expanded, max_frontier)

        expanded += 1
        for child in successors(state):
            if child in seen:
                continue
            seen.add(child)
            parent[child] = state
            frontier.append(child)

    return SearchResult([], float("inf"), expanded, max_frontier)


def astar(
    start: StateT,
    is_goal: Callable[[StateT], bool],
    successors: Callable[[StateT], Iterable[StateT]],
    step_cost: Callable[[StateT, StateT], float] = lambda _a, _b: 1.0,
    heuristic: Callable[[StateT], float] = lambda _s: 0.0,
) -> SearchResult[StateT]:
    """Run A* graph search with reopen-on-improvement semantics."""
    sequence = count()
    frontier: list[tuple[float, int, StateT]] = []
    heappush(frontier, (heuristic(start), next(sequence), start))

    parent: dict[StateT, StateT | None] = {start: None}
    g_score: dict[StateT, float] = {start: 0.0}
    closed: set[StateT] = set()
    expanded = 0
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        _priority, _seq, state = heappop(frontier)
        if state in closed:
            continue

        if is_goal(state):
            return SearchResult(
                _reconstruct_path(parent, state),
                g_score[state],
                expanded,
                max_frontier,
            )

        closed.add(state)
        expanded += 1

        for child in successors(state):
            tentative_g = g_score[state] + step_cost(state, child)
            if tentative_g >= g_score.get(child, float("inf")):
                continue

            parent[child] = state
            g_score[child] = tentative_g
            if child in closed:
                closed.remove(child)
            priority = tentative_g + heuristic(child)
            heappush(frontier, (priority, next(sequence), child))

    return SearchResult([], float("inf"), expanded, max_frontier)
