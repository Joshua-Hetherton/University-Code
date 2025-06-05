from tictactoe import Board, Player
from tictactoe_state_graph import TicTacToeState, TicTacToeStateGraph
from tictactoe_state_graph_writer import StateGraphWriter


if __name__ == "__main__":
    # Create a 3x3 board with half of the positions filled
    # (X plays first, then O, then X, etc.)
    #Question 16 -----------------------------------------------------------------
    board=Board(size=3)
    board=board.place_marker(0,0, Player.X)
    ##----------------------------------------------------------------------------
    
    ##Question 17-----------------------------------------------------------------
    # board=Board(size=4)
    # #Working from Left to right by row
    # #0
    # board=board.place_marker(0,2,Player.O)
    # board=board.place_marker(0,3,Player.X)
    # #1
    # board=board.place_marker(1,2,Player.O)
    # board=board.place_marker(1,3,Player.X)
    # #2
    # board=board.place_marker(2,0,Player.O)
    # board=board.place_marker(2,1,Player.X)
    # #3
    # board=board.place_marker(3,0,Player.O)
    # board=board.place_marker(3,1,Player.X)

    ##----------------------------------------------------------------------------
    ##TEST CASE-----------------------------------------------------
    # board = Board(size=3)  # Create an empty 3x3 board

    # Fill half of the positions (5 positions in a 3x3 board)
    # First move: X plays at the center
    # board = board.place_marker(1, 1, Player.X)
    # Second move: O plays at the top-left
    # board = board.place_marker(1, 1, Player.O)
    # Third move: X plays at the top-right
    # board = board.place_marker(0, 2, Player.X)
    # Fourth move: O plays at the bottom-left
    # board = board.place_marker(2, 0, Player.O)
    # Fifth move: X plays at the bottom-right
    # board = board.place_marker(2, 2, Player.X)
    #######-----------------------------------------------------------

        ##TEST CASE 2-----------------------------------------------------
    # board = Board(size=3)  # Create an empty 3x3 board

    # # Fill half of the positions (5 positions in a 3x3 board)
    # # First move: X plays at the center
    # board = board.place_marker(1, 1, Player.X)
    # board = board.place_marker(0,0, Player.O)
    
    # board = board.place_marker(2, 1, Player.X)
    # board = board.place_marker(0,2, Player.O)

    # board = board.place_marker(1,2, Player.X)
    # board = board.place_marker(2,2, Player.O)

    # board = board.place_marker(2,0, Player.X)

    #######-----------------------------------------------------------

    # Display the board state
    print("Current board state:")
    print(board)

    # Check if there's a winner already
    winner = board.evaluate_winner()
    if winner:
        print(f"Winner: {winner}")
    elif board.is_full():
        print("Game ended in a draw")
    else:
        print("Game is still in progress")
        print(f"Empty positions: {len(board.get_cells(Player.EMPTY))}")

    # Create a TicTacToeState from the board with O's turn next
    state = TicTacToeState(board=board, turn=Player.O)
    print("\nCurrent game state:")
    print(state)

    # Generate all possible next moves
    next_states = state.generate_successor_states()
    print(f"\nThere are {len(next_states)} possible next moves")

    # Create a state graph starting from the current state
    graph = TicTacToeStateGraph()
    graph.construct_from_initial_state(state)

    # Count the number of states in the graph
    total_states = graph.node_count
    print(f"Total number of game states from current position: {total_states}")

    # Count terminal states (game over states)
    terminal_states = graph.terminal_state_count
    print(f"Number of possible end game states: {terminal_states}")

    # Get statistics on game outcomes
    x_wins = len(graph.get_winning_states(Player.X))
    o_wins = len(graph.get_winning_states(Player.O))
    draws = len(graph.get_draw_states())
    print(f"Possible outcomes: X wins: {x_wins}, O wins: {o_wins}, Draws: {draws}")

    # Get degree statistics for the graph
    print("\nDegree distribution statistics:")
    degree_counts = graph.count_nodes_by_degree()

    print("In-degree distribution (number of ways to reach a state):")
    for degree, count in sorted(degree_counts["in_degree"].items()):
        print(f"  {degree}: {count} states")

    print("\nOut-degree distribution (number of possible moves from a state):")
    for degree, count in sorted(degree_counts["out_degree"].items()):
        print(f"  {degree}: {count} states")

    # Print statistics for the initial state
    print("\nInitial state statistics:")
    initial_in_degree = graph.get_in_degree(state)
    initial_out_degree = graph.get_out_degree(state)
    print(f"  In-degree: {initial_in_degree} (number of parent states)")
    print(f"  Out-degree: {initial_out_degree} (number of possible next moves)")

    # Get leaf states (terminal states)
    leaf_states = graph.get_leaf_states()
    print(f"\nLeaf states (states with no further moves): {len(leaf_states)}")

    # Calculate maximum depth of the game tree
    try:
        max_depth = graph.get_max_depth_from_root()
        print(f"Maximum depth from current state: {max_depth} moves")
    except Exception as e:
        print(f"Could not calculate maximum depth: {e}")

    # Write the state graph to files
    writer = StateGraphWriter(output_dir="game_states")
    writer.write_state_graph(graph)
    print("\nState graph written to directory: game_states")
