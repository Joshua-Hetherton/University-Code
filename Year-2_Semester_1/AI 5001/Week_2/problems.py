"""
This file defines the Problem class which is an abstract class for representing formal problems,
and the EightPuzzle class which is a specific implementation for the 8-puzzle game.

Code is adapted from AIMA Python code (https://github.com/aimacode/aima-python) for the purpose
of educational demonstration. The original code is licensed under the MIT License, and all copyright
belongs to the original authors. The MIT License is reproduced below.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numbers import Number  # For type hint/checking


class Problem:
    """
    The abstract class for a formal problem.

    For each specific problem, you should subclass this and implement the
    methods actions and result, and possibly __init__, goal_test, path_cost,
    and h. Then you will create instances of your subclass and solve them
    with the various search functions.
    """

    def __init__(self, initial_state, goal_state=None):
        """
        The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments.
        """
        self.initial_state = initial_state
        self.goal_state = goal_state

    def actions(self, state):
        """
        Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once.
        """
        raise NotImplementedError

    def result(self, state, action):
        """
        Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).
        """
        raise NotImplementedError

    def goal_test(self, state) -> bool:
        """
        Return True if the state is a goal. The default method compares the
        state to self.goal_state or checks for state in self.goal_state if it is a
        list, as specified in the constructor.
        """
        if isinstance(self.goal_state, list):
            return state in self.goal_state
        else:
            return state == self.goal_state

    def path_cost(self, c: Number, from_state, action, to_state) -> Number:
        """
        Return the cost of a solution path that arrives at to_state from
        from_state via action, assuming cost c to get up to from_state. The default
        method costs 1 for every step in the path.
        """
        return c + 1

    def h(self, state) -> Number:
        """
        Return the heuristic value for a given state. Default heuristic function used is
        h(n) = 0.
        """
        return 0

    def __same_state_type(self, state) -> bool:
        """Check if the state is equivalent"""
        return isinstance(state, type(self.initial_state))


class EightPuzzle(Problem):
    """
    The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where element at
    index i represents the tile number at index i (0 if it's an empty square)
    """

    def __init__(self, initial: tuple, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """Define goal state and initialize a problem"""
        super().__init__(initial, goal)

    def find_blank_square(self, state: tuple) -> int:
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state: tuple) -> list[str]:
        """
        Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment
        """

        possible_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove("LEFT")
        if index_blank_square < 3:
            possible_actions.remove("UP")
        if index_blank_square % 3 == 2:
            possible_actions.remove("RIGHT")
        if index_blank_square > 5:
            possible_actions.remove("DOWN")

        return possible_actions

    def result(self, state: tuple, action: str) -> tuple:
        """
        Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state
        """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {"UP": -3, "DOWN": 3, "LEFT": -1, "RIGHT": 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state: tuple) -> bool:
        """Given a state, return True if state is a goal state or False, otherwise"""

        return state == self.goal_state

    def check_solvability(self, state: tuple) -> bool:
        """Checks if the given state is solvable"""

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, state: tuple) -> int:
        """
        Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles
        """

        return sum(s != g for (s, g) in zip(state, self.goal_state))

class ThreePuzzle(Problem):
    """
    The problem of sliding tiles numbered from 1 to 3 on a 2x2 board, where one of the
    squares is a blank. A state is represented as a tuple of length 4, where element at
    index i represents the tile number at index i (0 if it's an empty square)
    """

    def __init__(self, initial: tuple, goal=(1, 2, 3, 0)):
        """Define goal state and initialize a problem"""
        super().__init__(initial, goal)

    def find_blank_square(self, state: tuple) -> int:
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state: tuple) -> list[str]:
        """
        Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment
        """

        possible_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 2 == 0:
            possible_actions.remove("LEFT")
        if index_blank_square < 2:
            possible_actions.remove("UP")
        if index_blank_square % 2 == 1:
            possible_actions.remove("RIGHT")
        if index_blank_square > 1:
            possible_actions.remove("DOWN")

        return possible_actions


    def result(self, state: tuple, action: str) -> tuple:
        """
        Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state
        """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {"UP": -2, "DOWN": 2, "LEFT": -1, "RIGHT": 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state: tuple) -> bool:
        """Given a state, return True if state is a goal state or False, otherwise"""

        return state == self.goal_state

    def check_solvability(self, state: tuple) -> bool:
        """Checks if the given state is solvable"""

        inversion = 0
        seq = [t for t in state if t != 0]
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                if seq[i] > seq[j]:
                    inversion += 1

        # For 2x2 (even width):
        blank_row_from_bottom = 2 - (self.find_blank_square(state) // 2)
        return (inversion + blank_row_from_bottom) % 2 == 0

    def h(self, state: tuple) -> int:
        """
        Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles
        """

        return sum(s != g for (s, g) in zip(state, self.goal_state))

class FifteenPuzzle(Problem):
    """
    The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where element at
    index i represents the tile number at index i (0 if it's an empty square)
    """

    def __init__(self, initial: tuple, goal=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)):
        """Define goal state and initialize a problem"""
        super().__init__(initial, goal)

    def find_blank_square(self, state: tuple) -> int:
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state: tuple) -> list[str]:
        """
        Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment
        """

        possible_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 4 == 0:
            possible_actions.remove("LEFT")
        if index_blank_square < 4:
            possible_actions.remove("UP")
        if index_blank_square % 4 == 3:
            possible_actions.remove("RIGHT")
        if index_blank_square > 11:
            possible_actions.remove("DOWN")

        return possible_actions

    def result(self, state: tuple, action: str) -> tuple:
        """
        Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state
        """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {"UP": -3, "DOWN": 3, "LEFT": -1, "RIGHT": 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state: tuple) -> bool:
        """Given a state, return True if state is a goal state or False, otherwise"""

        return state == self.goal_state

    def check_solvability(self, state: tuple) -> bool:
        """Checks if the given state is solvable"""

        inversion = 0
        seq = [t for t in state if t != 0]
        for i in range(len(seq)):
            for j in range(i+1, len(seq)):
                if seq[i] > seq[j]:
                    inversion += 1
        blank_row_from_bottom = self.size - (self.find_blank_square(state) // self.size)
        return (inversion + blank_row_from_bottom) % 2 == 0

    def h(self, state: tuple) -> int:
        """
        Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles
        """

        return sum(s != g for (s, g) in zip(state, self.goal_state))

class Generate_Tuple:
    """
    Class to generate a random tuple of given size
    """

    import random

    def __init__(self, size: int):
        self.size = size

    def generate(self, size) -> tuple:
        """Generate a random tuple of given size"""
        lst = list(range(size))
        self.random.shuffle(lst)
        return tuple(lst)

if __name__ == "__main__":
    # Example usage:
    type=int(input("""Pick Puzzle Type:
          1. 8-Puzzle
          2. 3-Puzzle\n"""))

    match type:
        case 1:
            initial_state = Generate_Tuple(9).generate(9)
            problem = EightPuzzle(initial_state)
        case 2:
            initial_state = Generate_Tuple(4).generate(4)
            problem = ThreePuzzle(initial_state)
    print("Initial State:")
    print(initial_state)
    print("Is the initial state solvable?", problem.check_solvability(initial_state))
    print("Possible actions from the initial state:", problem.actions(initial_state))
    new_state = problem.result(initial_state, "RIGHT")
    print("State after moving RIGHT:")
    print(new_state)
    print("Is the new state a goal state?", problem.goal_test(new_state))
    print(
        "Heuristic value (number of misplaced tiles) for the new state:",
        problem.h(new_state),
    )

