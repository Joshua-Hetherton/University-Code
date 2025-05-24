"""
tictactoe_state_graph_writer.py - Writes Tic-Tac-Toe states to text files

This module provides functionality to export a TicTacToe state graph to:
1. Individual text files representing each game state
2. An index file with categorized lists of states
3. A GraphViz DOT file for visual state graph representation

The StateGraphWriter class handles the entire process of converting an in-memory
state graph into a navigable file structure where each state file contains:
- The board representation
- Game status information (in progress, winner, draw)
- Links to possible next states (child nodes)
- Links to parent states
- State metrics (in-degree and out-degree)

Usage:
    writer = StateGraphWriter(output_dir="state_files")
    writer.write_state_graph(graph)  # Writes all states and index files
"""

__author__ = "Tin LEELAVIMOLSILP"
__version__ = "1.2.1"

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tictactoe_state_graph import TicTacToeState, TicTacToeStateGraph
import shutil
import time


class StateGraphWriter:
    """
    A class to write TicTacToe game states from a state graph to individual text files.

    This class processes a TicTacToeStateGraph and creates a directory structure containing:
    1. Individual text files for each state (state_N.txt)
    2. An index.txt file categorizing all states
    3. A state_graph.dot file for visualization with GraphViz

    Each state file includes:
    - Visual representation of the board
    - Current game status (in progress, winner, draw)
    - Number of parent and child states
    - Links to possible next moves with coordinates
    - Links to parent states that led to this state

    The class ensures uniqueness by assigning stable IDs to states and maintains
    consistent cross-references between state files.
    """

    def __init__(self, output_dir: str = "states"):
        """
        Initialize the StateGraphWriter.

        Args:
            output_dir: Directory where state files will be written. If the directory
                      exists, it will be purged and recreated. Default is "states".

        Attributes:
            output_dir: The target output directory path
            _state_ids: Dictionary mapping TicTacToeState objects to unique string IDs
            _next_id: Counter for generating sequential state IDs
            _total_processed: Count of states processed during write operations
            _start_time: Timestamp for performance monitoring
        """
        self.output_dir = output_dir
        self._state_ids: Dict[TicTacToeState, str] = {}
        self._next_id = 0
        self._total_processed = 0
        self._start_time = 0

    def _create_output_directory(self) -> Path:
        """
        Create the output directory if it doesn't exist, or purge it if it does.

        This method ensures we have a clean directory to write state files to.
        It follows these steps:
        1. If the directory exists, attempt to remove it and all its contents
        2. If removal fails (e.g., due to permissions), issue a warning and continue
        3. Create a new empty directory (or use the existing one if it couldn't be removed)

        Returns:
            Path object to the created directory

        Raises:
            IOError: If the directory couldn't be created after attempted cleanup

        Note:
            This is a destructive operation that will delete existing content
            in the output directory without prompting.
        """
        output_path = Path(self.output_dir)

        # If directory exists, remove it and its contents
        if output_path.exists():
            try:
                shutil.rmtree(output_path)
            except (PermissionError, OSError) as e:
                print(f"Warning: Could not remove existing directory: {e}")
                # Try to continue with existing directory

        # Create a fresh directory
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise IOError(f"Failed to create output directory {self.output_dir}: {e}")

        return output_path

    def _get_state_id(self, state: TicTacToeState) -> str:
        """
        Get a unique ID for a state, creating one if it doesn't exist.

        This method maintains a mapping between TicTacToeState objects and string IDs
        in the format "state_N" where N is a sequential number. Each unique state is
        assigned an ID the first time it's encountered, and that same ID is returned
        for all subsequent requests for the same state.

        The consistent mapping is crucial for maintaining proper cross-references
        between state files.

        Args:
            state: The game state to get an ID for

        Returns:
            A unique string ID for the state in the format "state_N"

        Note:
            State IDs are assigned sequentially based on the order in which states
            are first encountered, not based on any inherent property of the states.
        """
        if state not in self._state_ids:
            self._state_ids[state] = f"state_{self._next_id}"
            self._next_id += 1
        return self._state_ids[state]

    def _format_state_file_content(
        self, state: TicTacToeState, graph: TicTacToeStateGraph
    ) -> str:
        """
        Format the contents of a state file including the state representation,
        game status, and links to adjacent states.

        The formatted content includes:
        1. Visual board representation with player markers
        2. Game status (in progress or game over with outcome)
        3. Graph metrics (in-degree and out-degree)
        4. Links to possible next moves with their coordinates
        5. Links to parent states that can lead to this state

        The links are formatted with .txt extensions so they can be easily navigated
        when viewed in a file explorer or text editor.

        Args:
            state: The game state to format
            graph: The state graph containing the state

        Returns:
            Formatted content for the state file as a string

        Note:
            Graph metrics are enclosed in try-except blocks to gracefully handle
            states that may not be fully integrated into the graph.
        """
        content = []

        # Add state representation
        content.append(str(state))
        content.append("")

        # Add game status information
        if state.is_game_over():
            winner = state.evaluate_winner()
            if winner:
                content.append(f"GAME OVER - {winner} wins")
            else:
                content.append("GAME OVER - Draw")
        else:
            content.append(f"GAME IN PROGRESS - {state.turn}'s turn")

        # Add graph metrics for this state
        try:
            in_degree = graph.get_in_degree(state)
            out_degree = graph.get_out_degree(state)
            content.append(f"Parent states: {in_degree}")
            content.append(f"Child states: {out_degree}")
        except KeyError:
            content.append("State not in graph")

        # Add links to adjacent states (possible next moves)
        content.append("")
        content.append("Possible next moves:")

        try:
            next_states = graph.get_next_states(state)
            if next_states:
                for i, next_state in enumerate(next_states, 1):
                    next_id = self._get_state_id(next_state)
                    # Find which move was made to reach this state
                    move_pos = self._find_move_position(state, next_state)
                    if move_pos:
                        r, c = move_pos
                        content.append(f"  Move {i}: ({r},{c}) -> {next_id}.txt")
                    else:
                        content.append(f"  Move {i}: (unknown) -> {next_id}.txt")
            else:
                content.append("  None (terminal state)")
        except KeyError:
            content.append("  State not in graph")

        # Add links to parent states
        content.append("")
        content.append("Parent states (moves that led here):")
        parent_states = self._find_parent_states(state, graph)

        if parent_states:
            for i, parent_state in enumerate(parent_states, 1):
                parent_id = self._get_state_id(parent_state)
                content.append(f"  Parent {i}: {parent_id}.txt")
        else:
            content.append("  None (initial state)")

        return "\n".join(content)

    def _find_move_position(
        self, state: TicTacToeState, next_state: TicTacToeState
    ) -> Optional[Tuple[int, int]]:
        """
        Find the position where a move was made between two states.

        This method compares two sequential game states to determine which board position
        changed from empty to occupied, representing the move that was made.

        The method assumes that:
        - next_state is reachable from state in a single move
        - Exactly one empty position in state became occupied in next_state

        If these assumptions don't hold (e.g., states aren't sequential), the method
        may return None.

        Args:
            state: The original state
            next_state: The state after a move was made

        Returns:
            Tuple containing (row, col) of the move, or None if no move found

        Example:
            If state has an empty cell at (1,2) and next_state has an X at (1,2),
            this method will return (1,2).
        """
        for r in range(state.board.size):
            for c in range(state.board.size):
                if (
                    state.board.positions[r][c].name == "EMPTY"
                    and next_state.board.positions[r][c].name != "EMPTY"
                ):
                    return r, c
        return None

    def _find_parent_states(
        self, state: TicTacToeState, graph: TicTacToeStateGraph
    ) -> List[TicTacToeState]:
        """
        Find all parent states of a given state in the graph.

        A parent state is one that can reach the target state in a single move.
        This method performs a reverse lookup in the graph's adjacency structure
        to identify all states that have the target state as one of their next states.

        This is important for building the complete navigation structure between states,
        particularly for displaying "previous moves" or "moves that led here" in state files.

        Args:
            state: The state to find parents for
            graph: The state graph containing the state

        Returns:
            List of parent states that can reach the target state in one move

        Note:
            This method is computationally expensive for large graphs since it must
            check every state in the graph. For very large state spaces, a more
            efficient indexing structure would be beneficial.
        """
        parent_states = []
        for potential_parent in graph:
            try:
                next_states = graph.get_next_states(potential_parent)
                if state in next_states:
                    parent_states.append(potential_parent)
            except KeyError:
                continue
        return parent_states

    def write_dot_file(self, graph: TicTacToeStateGraph) -> str:
        """
        Generate a DOT representation of the state graph for visualization with GraphViz.

        The DOT file uses color coding to distinguish different types of states:
        - Initial state: Light blue
        - X wins: Light green
        - O wins: Light coral (reddish)
        - Draws: Light yellow
        - Other states: Default white

        Edges between states are labeled with the move coordinates (row,col) that led
        from one state to the next.

        Args:
            graph: The state graph to convert to DOT format

        Returns:
            Path to the generated DOT file

        Raises:
            IOError: If writing the file fails

        Notes:
            To convert the DOT file to a visual graph image, you can install and use GraphViz.
            Other options include using online DOT file viewers or other graph visualization tools.
            Such online DOT viewers include:
            - https://dreampuf.github.io/GraphvizOnline/
            - https://viz-js.com/
        """
        # Create the DOT file path
        dot_file = Path(self.output_dir) / "state_graph.dot"

        # Start building the DOT content
        dot_content = ["digraph TicTacToeStateGraph {"]
        dot_content.append("  // Graph styling")
        dot_content.append('  graph [rankdir=LR, fontname="Arial", bgcolor="white"];')
        dot_content.append('  node [shape=record, fontname="Arial", fontsize=10];')
        dot_content.append('  edge [fontname="Arial", fontsize=9];')
        dot_content.append("")

        # Add node definitions with styling based on state type
        dot_content.append("  // Node definitions")
        for state in graph:
            state_id = self._get_state_id(state)

            # Create a label with the board representation
            board_str = str(state.board).replace("\n", "\\n")
            label = f"{state_id}\\n{board_str}\\nTurn: {state.turn}"

            # Style based on game state
            attrs = []

            try:
                if graph.get_in_degree(state) == 0:
                    # Initial state
                    attrs.append("style=filled")
                    attrs.append("fillcolor=lightblue")
                elif state.is_game_over():
                    winner = state.evaluate_winner()
                    if winner and winner.name == "X":
                        attrs.append("style=filled")
                        attrs.append("fillcolor=lightgreen")
                        label += "\\nX wins"
                    elif winner and winner.name == "O":
                        attrs.append("style=filled")
                        attrs.append("fillcolor=lightcoral")
                        label += "\\nO wins"
                    else:
                        attrs.append("style=filled")
                        attrs.append("fillcolor=lightyellow")
                        label += "\\nDraw"
            except KeyError:
                # Just use default style if there's an error computing node degree
                pass

            # Add the node definition
            node_def = f'  {state_id} [label="{label}"'
            if attrs:
                node_def += ", " + ", ".join(attrs)
            node_def += "];"
            dot_content.append(node_def)

        dot_content.append("")
        dot_content.append("  // Edge definitions")

        # Add edge definitions
        for state in graph:
            state_id = self._get_state_id(state)
            try:
                for next_state in graph.get_next_states(state):
                    next_id = self._get_state_id(next_state)

                    # Find which move was made
                    move_pos = self._find_move_position(state, next_state)
                    move_label = ""
                    if move_pos:
                        r, c = move_pos
                        move_label = f"({r},{c})"

                    # Add the edge with the move as a label
                    edge_def = f'  {state_id} -> {next_id} [label="{move_label}"];'
                    dot_content.append(edge_def)
            except KeyError:
                # Skip edges for nodes not in the graph
                continue

        # Close the graph
        dot_content.append("}")

        # Write to file with error handling
        try:
            with open(dot_file, "w") as f:
                f.write("\n".join(dot_content))
        except IOError as e:
            raise IOError(f"Failed to write DOT file to {dot_file}: {e}")

        return str(dot_file)

    def write_state_graph(self, graph: TicTacToeStateGraph) -> None:
        """
        Write all states from a state graph to individual text files.

        This is the main method that orchestrates the entire export process:
        1. Creates/resets the output directory
        2. Processes each state and writes it to a separate file
        3. Creates an index file listing all states by category
        4. Generates a DOT file for GraphViz visualization

        The method provides progress updates during processing, which can be
        particularly helpful for large graphs. Performance statistics are also
        reported when processing completes.

        Args:
            graph: The state graph to write to files

        Raises:
            IOError: If creating directories or writing files fails

        Note:
            This method will overwrite any existing files in the output directory.
        """
        # Start timing for performance monitoring
        self._start_time = time.time()
        self._total_processed = 0

        print(f"Starting to process {graph.node_count} states...")

        # Create output directory
        output_path = self._create_output_directory()

        # Reset state IDs and counter
        self._state_ids = {}
        self._next_id = 0

        # Get total count for progress reporting
        total_states = len(graph._adjacency_list)

        # Write each state to a separate file
        for i, state in enumerate(graph):
            # Update progress every 10% or at least every 25 nodes
            if i % max(1, min(total_states // 10, 25)) == 0:
                elapsed = time.time() - self._start_time
                if elapsed > 0:
                    nodes_per_sec = i / elapsed
                    remaining = (
                        (total_states - i) / nodes_per_sec if nodes_per_sec > 0 else 0
                    )
                    print(
                        f"Progress: {i}/{total_states} states processed "
                        f"({i / total_states * 100:.1f}%) - "
                        f"ETA: {remaining:.1f}s"
                    )

            state_id = self._get_state_id(state)
            filename = output_path / f"{state_id}.txt"

            content = self._format_state_file_content(state, graph)

            try:
                with open(filename, "w") as f:
                    f.write(content)
            except IOError as e:
                print(f"Warning: Failed to write state file {filename}: {e}")

            self._total_processed += 1

        # Create an index file listing all states
        self._write_index_file(graph)

        # Generate DOT file for GraphViz visualization
        try:
            dot_file = self.write_dot_file(graph)
            print(f"Generated GraphViz DOT file: {dot_file}")
        except IOError as e:
            print(f"Warning: Failed to generate DOT file: {e}")

        elapsed = time.time() - self._start_time
        print(
            f"Completed processing {self._total_processed} states in {elapsed:.2f} seconds "
            f"({self._total_processed / elapsed:.1f} states/second)"
        )

    def _write_index_file(self, graph: TicTacToeStateGraph) -> None:
        """
        Write an index file listing all states with their IDs and brief descriptions.

        The index file contains:
        1. A summary section with total state counts and percentages
        2. The initial state(s)
        3. In-progress states sorted by ID
        4. Terminal states categorized by outcome (X wins, O wins, draws)

        This serves as a central directory to navigate the state files.

        Args:
            graph: The state graph to index

        Raises:
            IOError: If writing the file fails

        Note:
            The method uses the _categorize_states helper to efficiently group states.
        """
        filename = Path(self.output_dir) / "index.txt"

        # Create categorized state lists once to avoid repeated traversals
        categorized_states = self._categorize_states(graph)

        initial_states = categorized_states["initial"]
        in_progress_states = categorized_states["in_progress"]
        win_states_x = categorized_states["x_wins"]
        win_states_o = categorized_states["o_wins"]
        draw_states = categorized_states["draws"]

        # Add summary section
        content = ["GRAPH SUMMARY"]
        content.append(f"Total states: {graph.node_count}")
        terminal_count = len(win_states_x) + len(win_states_o) + len(draw_states)
        non_terminal_count = len(initial_states) + len(in_progress_states)

        # Validate counts with graph metrics
        if hasattr(graph, "terminal_state_count") and hasattr(
            graph, "non_terminal_state_count"
        ):
            terminal_count = graph.terminal_state_count
            non_terminal_count = graph.non_terminal_state_count

        content.append(
            f"Terminal states: {terminal_count} ({(terminal_count / graph.node_count * 100):.1f}% of total)"
        )
        content.append(
            f"Non-terminal states: {non_terminal_count} ({(non_terminal_count / graph.node_count * 100):.1f}% of total)"
        )

        # Write initial states
        content.append("\nINITIAL STATE:")
        content.append("----------------")
        if initial_states:
            for state_id, state in initial_states:
                content.append(f"{state_id}.txt - {state.turn}'s turn")
        else:
            content.append("None")

        # Write in-progress states
        content.append("\nIN-PROGRESS STATES:")
        content.append("-------------------")
        if in_progress_states:
            content.append(f"Count: {len(in_progress_states)}")
            for state_id, state in in_progress_states:
                content.append(f"{state_id}.txt - {state.turn}'s turn")
        else:
            content.append("None")

        # Write terminal states: X wins
        content.append(f"\nX WINS STATES (Count: {len(win_states_x)}):")
        content.append("-----------------------------------")
        if win_states_x:
            for state_id, state in win_states_x:
                content.append(f"{state_id}.txt")
        else:
            content.append("None")

        # Write terminal states: O wins
        content.append(f"\nO WINS STATES (Count: {len(win_states_o)}):")
        content.append("-----------------------------------")
        if win_states_o:
            for state_id, state in win_states_o:
                content.append(f"{state_id}.txt")
        else:
            content.append("None")

        # Write terminal states: Draws
        content.append(f"\nDRAW STATES (Count: {len(draw_states)}):")
        content.append("----------------------------------")
        if draw_states:
            for state_id, state in draw_states:
                content.append(f"{state_id}.txt")
        else:
            content.append("None")

        try:
            with open(filename, "w") as f:
                f.write("\n".join(content))
        except IOError as e:
            print(f"Warning: Failed to write index file {filename}: {e}")

    def _categorize_states(
        self, graph: TicTacToeStateGraph
    ) -> Dict[str, List[Tuple[str, TicTacToeState]]]:
        """
        Categorize all states in the graph by type for efficient index creation.

        This helper method organizes states into five distinct categories:
        - "initial": States with no parent states (in-degree of 0)
        - "in_progress": Non-terminal states with at least one parent
        - "x_wins": Terminal states where Player X has won
        - "o_wins": Terminal states where Player O has won
        - "draws": Terminal states that ended in a draw

        The states within each category are sorted by their assigned IDs to ensure
        consistent ordering across different runs.

        Args:
            graph: The state graph to analyze

        Returns:
            Dictionary with categorized states as {category: [(state_id, state), ...]}
        """
        # Create empty lists for each category
        categorized = {
            "initial": [],
            "in_progress": [],
            "x_wins": [],
            "o_wins": [],
            "draws": [],
        }

        # Sort states once by ID to ensure consistent ordering
        sorted_states = sorted(self._state_ids.keys(), key=lambda s: self._state_ids[s])

        for state in sorted_states:
            state_id = self._state_ids[state]
            if state.is_game_over():
                winner = state.evaluate_winner()
                if winner is not None and winner.name == "X":
                    categorized["x_wins"].append((state_id, state))
                elif winner is not None and winner.name == "O":
                    categorized["o_wins"].append((state_id, state))
                else:
                    categorized["draws"].append((state_id, state))
            else:
                try:
                    if graph.get_in_degree(state) == 0:
                        categorized["initial"].append((state_id, state))
                    else:
                        categorized["in_progress"].append((state_id, state))
                except KeyError:
                    # If we can't determine in-degree, assume it's in-progress
                    categorized["in_progress"].append((state_id, state))

        return categorized


if __name__ == "__main__":
    """
    Example usage of the StateGraphWriter class.
    
    This example demonstrates:
    1. Creating a partially filled Tic-Tac-Toe board
    2. Constructing a complete state graph from that board position
    3. Writing all states and visualization files to a directory
    
    The board configuration used is:
    X | O | _
    ---------
    _ | X | _
    ---------
    O | X | O
    
    with O's turn to play next.
    """
    # Example: Create a simple state graph with the partially filled board
    # and write it to files
    from tictactoe import Board, Player
    from tictactoe_state_graph import TicTacToeState, TicTacToeStateGraph

    # Create a partially filled board with some X and O markers
    # X | O |
    # ---------
    #   | X |
    # ---------
    # O | X | O

    # Initialize an empty board first
    board = Board()

    # Place markers at specific positions
    board = board.place_marker(0, 1, Player.O)  # O at top-middle
    board = board.place_marker(0, 0, Player.X)  # X at top-left
    board = board.place_marker(2, 0, Player.O)  # O at bottom-left
    board = board.place_marker(1, 1, Player.X)  # X at center
    board = board.place_marker(2, 2, Player.O)  # O at bottom-right
    board = board.place_marker(2, 1, Player.X)  # X at bottom-middle

    # Create a state with this board, assuming it's O's turn next
    state = TicTacToeState(board, Player.O)
    print("Initial partially filled state:")
    print(state)
    print()

    # Create a state graph for this game state
    graph = TicTacToeStateGraph()

    # Populate the graph
    graph.construct_from_initial_state(state)

    # Create a StateGraphWriter and write the graph to files
    writer = StateGraphWriter(output_dir="state_files")
    writer.write_state_graph(graph)
    print("Wrote state graph to state_files directory")
    print(f"Total states in graph: {len(graph._adjacency_list)}")
