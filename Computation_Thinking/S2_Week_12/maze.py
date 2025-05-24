"""
maze.py

A module that implements maze generation and solving algorithms.
Creates 2D mazes and solves them using various search algorithms.
"""
from enum import Enum, auto
from typing import List, NamedTuple, Optional, Callable, Dict, Tuple
import random
from dataclasses import dataclass
import time

# Import search algorithms from our generic search module
from generic_search import Node, dfs, bfs, node_to_path


class Cell(str, Enum):
    """
    Enum representing the different types of cells in a maze.
    Inherits from str to allow easy printing of the maze.
    """
    EMPTY = " "   # Traversable empty space
    BLOCKED = "X"  # Wall or obstacle
    START = "S"    # Starting position
    GOAL = "G"     # Goal position
    PATH = "*"     # Part of the solution path


class MazeLocation(NamedTuple):
    """
    Represents a position in the maze using row and column indices.
    Using NamedTuple for immutability and easy comparison.
    """
    row: int
    column: int
    
    def __str__(self) -> str:
        """String representation of the location"""
        return f"({self.row}, {self.column})"


class Direction(Enum):
    """Represents possible movement directions in the maze"""
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()
    
    @staticmethod
    def get_movement(direction: 'Direction') -> Tuple[int, int]:
        """
        Returns the row and column delta for a given direction
        
        Args:
            direction: The direction to move
            
        Returns:
            A tuple of (row_delta, column_delta)
        """
        if direction == Direction.UP:
            return (-1, 0)
        elif direction == Direction.RIGHT:
            return (0, 1)
        elif direction == Direction.DOWN:
            return (1, 0)
        elif direction == Direction.LEFT:
            return (0, -1)
        else:
            raise ValueError(f"Unknown direction: {direction}")


@dataclass
class MazeStats:
    """Records statistics about maze solving attempts"""
    algorithm_name: str
    path_length: int
    nodes_explored: int
    time_taken: float
    success: bool
    
    def __str__(self) -> str:
        """String representation of the maze solving statistics"""
        status = "Success" if self.success else "Failed"
        return (
            f"{self.algorithm_name}: {status}\n"
            f"  Path length: {self.path_length}\n"
            f"  Nodes explored: {self.nodes_explored}\n"
            f"  Time taken: {self.time_taken:.6f} seconds"
        )


class Maze:
    """
    Represents a 2D maze with walls, a start position, and a goal position.
    Provides methods to generate, display, and navigate the maze.
    """
    
    def __init__(self, rows: int = 10, columns: int = 10, sparseness: float = 0.2, 
                 start: Optional[MazeLocation] = None, 
                 goal: Optional[MazeLocation] = None) -> None:
        """
        Initialize a new maze with the specified dimensions and parameters.
       
        Args:
            rows: Number of rows in the maze (default: 10)
            columns: Number of columns in the maze (default: 10)
            sparseness: Probability of each cell being blocked (default: 0.2)
            start: Starting position (default: top-left corner if None)
            goal: Goal position (default: bottom-right corner if None)
            
        Raises:
            ValueError: If rows or columns are less than 2
            ValueError: If sparseness is not between 0 and 1
        """
        # Validate input parameters
        if rows < 2 or columns < 2:
            raise ValueError("Maze must have at least 2 rows and 2 columns")
        
        if not 0 <= sparseness <= 1:
            raise ValueError("Sparseness must be between 0 and 1")
            
        # Initialize basic instance variables
        self._rows: int = rows
        self._columns: int = columns
        
        # Set default start and goal if not provided
        self.start: MazeLocation = start if start is not None else MazeLocation(0, 0)
        self.goal: MazeLocation = goal if goal is not None else MazeLocation(rows - 1, columns - 1)
        
        # Validate start and goal positions
        if not self._is_valid_location(self.start) or not self._is_valid_location(self.goal):
            raise ValueError("Start and goal positions must be within maze boundaries")
        
        # Fill the grid with empty cells
        self._grid: List[List[Cell]] = [[Cell.EMPTY for _ in range(columns)] for _ in range(rows)]
        
        # Populate the grid with blocked cells
        self._randomly_fill(sparseness)
        
        # Fill the start and goal locations
        self._grid[self.start.row][self.start.column] = Cell.START
        self._grid[self.goal.row][self.goal.column] = Cell.GOAL
        
        # Ensure start and goal are not blocked
        self._ensure_path_exists()

    def _is_valid_location(self, location: MazeLocation) -> bool:
        """
        Check if a location is within the maze boundaries.
        
        Args:
            location: The location to check
            
        Returns:
            True if the location is valid, False otherwise
        """
        return (0 <= location.row < self._rows and
                0 <= location.column < self._columns)

    def _randomly_fill(self, sparseness: float) -> None:
        """
        Randomly fills the maze with blocked cells based on the sparseness parameter.
        
        Args:
            sparseness: Probability (0.0 to 1.0) of each cell being blocked
        """
        for row in range(self._rows):
            for column in range(self._columns):
                # Skip start and goal positions
                if MazeLocation(row, column) in (self.start, self.goal):
                    continue
                    
                # For each cell, generate a random number and compare with sparseness
                if random.random() < sparseness:
                    self._grid[row][column] = Cell.BLOCKED
                    
    def _ensure_path_exists(self) -> None:
        """
        Makes sure there is at least one path from start to goal.
        If not, it clears some blocked cells to create a path.
        """
        # Try to find a path using BFS
        if self.solve_bfs() is None:
            # No path exists, so clear some cells
            self._clear_path_between(self.start, self.goal)
            
    def _clear_path_between(self, start: MazeLocation, end: MazeLocation) -> None:
        """
        Creates a simple path between start and end by clearing blocked cells.
        Uses a simple approach: first move horizontally, then vertically.
        
        Args:
            start: Starting location
            end: Ending location
        """
        # Start at the beginning
        current_row, current_col = start.row, start.column
        
        # First move horizontally until we reach the goal column
        while current_col != end.column:
            step = 1 if current_col < end.column else -1
            current_col += step
            # Clear the cell if it's blocked
            if self._grid[current_row][current_col] == Cell.BLOCKED:
                self._grid[current_row][current_col] = Cell.EMPTY
                
        # Then move vertically until we reach the goal row
        while current_row != end.row:
            step = 1 if current_row < end.row else -1
            current_row += step
            # Clear the cell if it's blocked
            if self._grid[current_row][current_col] == Cell.BLOCKED:
                self._grid[current_row][current_col] = Cell.EMPTY
                
        # Make sure the goal is set correctly
        self._grid[end.row][end.column] = Cell.GOAL

    def __str__(self) -> str:
        """
        Returns a string representation of the maze for display.
        
        Returns:
            A multiline string where each character represents a cell
        """
        # Add top border
        output = "+" + "-" * self._columns + "+\n"
        
        # Add maze content with left and right borders
        for row in self._grid:
            output += "|" + "".join(c.value for c in row) + "|\n"
            
        # Add bottom border
        output += "+" + "-" * self._columns + "+"
        return output
        
    def __repr__(self) -> str:
        """Detailed representation of the maze for debugging"""
        return f"Maze({self._rows}x{self._columns}, start={self.start}, goal={self.goal})"

    def goal_test(self, ml: MazeLocation) -> bool:
        """
        Tests if the given location is the goal location.
        
        Args:
            ml: The location to test
            
        Returns:
            True if the location is the goal, False otherwise
        """
        return ml == self.goal

    def successors(self, ml: MazeLocation) -> List[MazeLocation]:
        """
        Finds all valid adjacent locations from the current location.
        
        Args:
            ml: The current location
            
        Returns:
            A list of valid adjacent locations (not blocked and within bounds)
        """
        locations: List[MazeLocation] = []
        
        # Check all four possible directions
        for direction in Direction:
            row_delta, col_delta = Direction.get_movement(direction)
            new_row, new_col = ml.row + row_delta, ml.column + col_delta
            new_location = MazeLocation(new_row, new_col)
            
            # Check if the new location is valid and not blocked
            if (self._is_valid_location(new_location) and
                self._grid[new_row][new_col] != Cell.BLOCKED):
                locations.append(new_location)
                
        return locations

    def mark(self, path: List[MazeLocation]) -> None:
        """
        Marks a path on the maze with '*' characters.
        
        Args:
            path: List of locations forming the path from start to goal
        """
        # Mark each location in the path (except start and goal)
        for maze_location in path:
            if maze_location != self.start and maze_location != self.goal:
                self._grid[maze_location.row][maze_location.column] = Cell.PATH
            
        # Ensure start and goal remain marked correctly
        self._grid[self.start.row][self.start.column] = Cell.START
        self._grid[self.goal.row][self.goal.column] = Cell.GOAL
    
    def clear(self, path: List[MazeLocation]) -> None:
        """
        Clears a previously marked path from the maze.
        
        Args:
            path: List of locations to clear
        """
        # Clear each location in the path (except start and goal)
        for maze_location in path:
            if maze_location != self.start and maze_location != self.goal:
                self._grid[maze_location.row][maze_location.column] = Cell.EMPTY
            
        # Ensure start and goal remain marked correctly  
        self._grid[self.start.row][self.start.column] = Cell.START
        self._grid[self.goal.row][self.goal.column] = Cell.GOAL
        
    def solve_dfs(self) -> Optional[List[MazeLocation]]:
        """
        Solve the maze using Depth-First Search.
        
        Returns:
            A list of locations from start to goal, or None if no solution exists
        """
        solution_node = dfs(self.start, self.goal_test, self.successors)
        if solution_node is None:
            return None
        return node_to_path(solution_node)
        
    def solve_bfs(self) -> Optional[List[MazeLocation]]:
        """
        Solve the maze using Breadth-First Search.
        
        Returns:
            A list of locations from start to goal, or None if no solution exists
        """
        solution_node = bfs(self.start, self.goal_test, self.successors)
        if solution_node is None:
            return None
        return node_to_path(solution_node)
    
    def solve_with_stats(self, algorithm_name: str, 
                        algorithm: Callable[[MazeLocation, Callable[[MazeLocation], bool], 
                                            Callable[[MazeLocation], List[MazeLocation]]], 
                                            Optional[Node[MazeLocation]]]) -> MazeStats:
        """
        Solve the maze using the specified algorithm and record statistics.
        
        Args:
            algorithm_name: Name of the algorithm for display
            algorithm: The search algorithm function to use
            
        Returns:
            MazeStats object containing statistics about the solving process
        """
        # Track the number of explored nodes
        explored_count = 0
        original_successors = self.successors
        
        # Wrap the successors function to count explored nodes
        def counting_successors(ml: MazeLocation) -> List[MazeLocation]:
            nonlocal explored_count
            successors_list = original_successors(ml)
            explored_count += len(successors_list)
            return successors_list
        
        # Temporarily replace the successors function
        self.successors = counting_successors
        
        # Measure the time taken to solve
        start_time = time.time()
        solution_node = algorithm(self.start, self.goal_test, self.successors)
        end_time = time.time()
        
        # Restore the original successors function
        self.successors = original_successors
        
        # Calculate statistics
        success = solution_node is not None
        path = node_to_path(solution_node) if success else []
        path_length = len(path) - 1 if success else 0  # -1 because we count edges, not nodes
        
        return MazeStats(
            algorithm_name=algorithm_name,
            path_length=path_length,
            nodes_explored=explored_count,
            time_taken=end_time - start_time,
            success=success
        )


# Demo code for maze generation and solving
if __name__ == "__main__":

    # Allow user to specify maze parameters or use defaults
    try:
        rows = int(input("Enter number of rows (default 10): ") or "10")
        cols = int(input("Enter number of columns (default 10): ") or "10")
        sparseness = float(input("Enter sparseness between 0.0 and 1.0 (default 0.2): ") or "0.2")
        
        # Create a new random maze
        m = Maze(rows=rows, columns=cols, sparseness=sparseness)
        print("\nRandom Maze:")
        print(m)
        
        # Collect solving statistics for each algorithm
        stats = []
        
        # Test Depth-First Search
        print("\nSolving with Depth-First Search...")
        dfs_stats = m.solve_with_stats("DFS", dfs)
        stats.append(dfs_stats)
        
        if dfs_stats.success:
            # Get the solution path
            path1 = m.solve_dfs()
            m.mark(path1)
            print("DFS Solution:")
            print(m)
            
            # Clear the path for the next algorithm
            m.clear(path1)
        else:
            print("No solution found using depth-first search!")
        
        # Test Breadth-First Search
        print("\nSolving with Breadth-First Search...")
        bfs_stats = m.solve_with_stats("BFS", bfs)
        stats.append(bfs_stats)
        
        if bfs_stats.success:
            # Get the solution path
            path2 = m.solve_bfs()
            m.mark(path2)
            print("BFS Solution:")
            print(m)
        else:
            print("No solution found using breadth-first search!")
        
        # Print comparison of statistics
        print("\nAlgorithm Comparison:")
        for stat in stats:
            print(stat)
            print()
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Using default maze parameters instead.")
        
        m = Maze()
        print("\nRandom Maze:")
        print(m)
        
        # Test both search algorithms
        print("\nSolving with Depth-First Search...")
        path1 = m.solve_dfs()
        
        if path1:
            m.mark(path1)
            print("DFS Solution:")
            print(m)
            m.clear(path1)
        else:
            print("No solution found using depth-first search!")
        
        print("\nSolving with Breadth-First Search...")
        path2 = m.solve_bfs()
        
        if path2:
            m.mark(path2)
            print("BFS Solution:")
            print(m)
        else:
            print("No solution found using breadth-first search!")