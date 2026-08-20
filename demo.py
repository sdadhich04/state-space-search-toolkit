from src.state_space_search.examples import demo_eight_puzzle, demo_river_crossing


def main() -> None:
    print("River crossing with BFS")
    river = demo_river_crossing()
    print(f"  states: {len(river.path)}")
    print(f"  expanded: {river.expanded}")
    print(f"  path: {river.path}")

    print("\nEight puzzle with A*")
    puzzle = demo_eight_puzzle()
    print(f"  moves: {len(puzzle.path) - 1}")
    print(f"  cost: {puzzle.cost}")
    print(f"  expanded: {puzzle.expanded}")
    for state in puzzle.path:
        print(f"  {state}")


if __name__ == "__main__":
    main()
