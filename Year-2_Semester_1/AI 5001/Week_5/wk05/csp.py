"""CSP (Constraint Satisfaction Problems) problems and solvers. (Chapter 6)

This module implements various algorithms for solving Constraint Satisfaction Problems (CSPs).
A CSP consists of:
- Variables: Things we need to assign values to (e.g., calendar time slots, map regions)
- Domains: Possible values for each variable (e.g., times, colors)
- Constraints: Rules that restrict which combinations of values are allowed

Common CSP examples:
- Map coloring: Color adjacent regions with different colors
- N-Queens: Place N queens on a chessboard so none attack each other
- Sudoku: Fill a 9x9 grid with digits 1-9 following specific rules
- Class scheduling: Assign classes to time slots without conflicts
"""

import itertools  # For creating combinations and permutations
import random  # For random choices in local search algorithms
import re  # For regular expression parsing (used in Sudoku)
import string  # For string constants (used in crossword puzzles)
from collections import defaultdict, Counter  # Efficient data structures
from functools import reduce  # For functional programming operations
from operator import eq, neg  # Operator functions for cleaner code

from sortedcontainers import SortedSet  # Efficient sorted container for arc consistency

import search  # General search algorithms framework
from utils import argmin_random_tie, count, first, extend  # Utility functions


class CSP(search.Problem):
    """A Constraint Satisfaction Problem (CSP) - the main class for representing CSPs.

    A CSP consists of three components:
    1. VARIABLES: A list of variables that need values (e.g., ['WA', 'NT', 'SA'] for regions)
    2. DOMAINS: A dictionary mapping each variable to its possible values
                (e.g., {'WA': ['red', 'green', 'blue'], 'NT': ['red', 'green', 'blue']})
    3. CONSTRAINTS: Rules about which variable-value combinations are allowed
                   (e.g., adjacent regions must have different colors)

    Example - Simple Map Coloring:
        variables = ['WA', 'NT']  # Western Australia, Northern Territory
        domains = {'WA': ['red', 'blue'], 'NT': ['red', 'blue']}
        neighbors = {'WA': ['NT'], 'NT': ['WA']}  # WA and NT are adjacent
        constraints = lambda A, a, B, b: a != b  # Different colors for neighbors

    The CSP class provides:
    - Basic operations: assign(), unassign(), nconflicts()
    - Search interface: actions(), result(), goal_test()
    - Constraint propagation support: support_pruning(), prune(), restore()

    Key data structures:
    - variables: List of all variables in the problem
    - domains: Original domains for each variable (never modified)
    - neighbors: Dictionary showing which variables constrain each other
    - constraints: Function that checks if two variables can have certain values
    - curr_domains: Current domains after constraint propagation (can shrink)
    - nassigns: Counter tracking how many assignments we've made (for analysis)
    """

    def __init__(self, variables, domains, neighbors, constraints):
        """Create a new CSP instance.

        Args:
            variables: List of variable names (if None, uses domains.keys())
            domains: Dict mapping each variable to list of possible values
            neighbors: Dict mapping each variable to list of variables it constrains with
            constraints: Function that returns True if two variables can have given values
                        Format: constraints(var1, val1, var2, val2) -> bool

        Example:
            # Simple 2-variable map coloring
            CSP(['A', 'B'],
                {'A': ['red', 'blue'], 'B': ['red', 'blue']},
                {'A': ['B'], 'B': ['A']},
                lambda var1, val1, var2, val2: val1 != val2)
        """
        super().__init__(())  # Initialize parent search.Problem class
        variables = variables or list(
            domains.keys()
        )  # Use domains.keys() if no variables given
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = (
            None  # Will be initialized when needed for constraint propagation
        )
        self.nassigns = (
            0  # Track number of assignments made (useful for performance analysis)
        )

    def assign(self, var, val, assignment):
        """Assign a value to a variable in the current assignment.

        This is more than just assignment[var] = val because we also:
        - Track the number of assignments made (for performance analysis)
        - Allow subclasses to do additional bookkeeping (like conflict tracking)

        Args:
            var: Variable to assign
            val: Value to assign to the variable
            assignment: Current assignment dictionary (modified in place)
        """
        assignment[var] = val
        self.nassigns += 1  # Keep track of how many assignments we've made

    def unassign(self, var, assignment):
        """Remove a variable assignment during backtracking.

        Important: Only use this when backtracking! If you want to change
        a variable to a new value, just call assign() - don't unassign first.

        Args:
            var: Variable to remove from assignment
            assignment: Current assignment dictionary (modified in place)
        """
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        """Count how many conflicts var=val would have with current assignments.

        A conflict occurs when:
        1. The other variable is already assigned a value, AND
        2. The constraint between var=val and other_var=other_val is violated

        This is a key function used by search algorithms to evaluate moves.

        Args:
            var: Variable we're considering assigning
            val: Value we're considering for that variable
            assignment: Current partial assignment

        Returns:
            int: Number of currently assigned variables that would conflict

        Example:
            If we have variables A, B, C with A=red, B=blue already assigned,
            and we're considering C=red, this counts conflicts with A and B.
        """

        # Helper function to check if a specific variable conflicts
        def conflict(var2):
            # Only count conflicts with variables that are already assigned
            return var2 in assignment and not self.constraints(
                var, val, var2, assignment[var2]
            )

        # Count conflicts across all neighboring variables
        return count(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the current assignment.

        Subclasses often override this to show prettier, problem-specific displays.
        For example, N-Queens shows a chessboard, Sudoku shows a grid.
        """
        print(assignment)

    # === Methods for integrating with general search algorithms ===
    # These methods allow CSPs to work with the search.py framework

    def actions(self, state):
        """Return possible actions from current state for general search algorithms.

        An action is (variable, value) pair for an unassigned variable that
        doesn't conflict with current assignments.

        Args:
            state: Current state (tuple of (var, val) pairs)

        Returns:
            List of (var, val) actions that don't cause immediate conflicts
        """
        if len(state) == len(self.variables):
            return []  # All variables assigned - no more actions possible
        else:
            assignment = dict(state)  # Convert state to assignment dictionary
            # Find first unassigned variable
            var = first([v for v in self.variables if v not in assignment])
            # Return all non-conflicting values for this variable
            return [
                (var, val)
                for val in self.domains[var]
                if self.nconflicts(var, val, assignment) == 0
            ]

    def result(self, state, action):
        """Apply an action to get a new state.

        Args:
            state: Current state (tuple of (var, val) pairs)
            action: Action to apply (var, val) pair

        Returns:
            New state with the action applied
        """
        (var, val) = action
        return state + ((var, val),)  # Add new assignment to state tuple

    def goal_test(self, state):
        """Check if we have a complete, valid solution.

        A solution is complete when:
        1. All variables are assigned values
        2. No constraints are violated

        Args:
            state: Current state to test

        Returns:
            bool: True if state is a valid complete solution
        """
        assignment = dict(state)
        return len(assignment) == len(self.variables) and all(
            self.nconflicts(variables, assignment[variables], assignment) == 0
            for variables in self.variables
        )

    # === Constraint Propagation Methods ===
    # These methods support advanced CSP solving techniques that eliminate
    # impossible values from variable domains before/during search

    def support_pruning(self):
        """Initialize data structures needed for constraint propagation.

        This creates curr_domains - a copy of domains that we can modify
        during search. We only create this when needed to save memory.

        Think of domains as the "original rules" and curr_domains as
        "what's still possible given what we've learned so far."
        """
        if self.curr_domains is None:
            # Create a modifiable copy of the original domains
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        """Temporarily assume var=value and track what we remove from other domains.

        This is used in backtracking search: when we try assigning var=value,
        we eliminate all other values from var's domain and keep track of
        what we removed so we can restore it if this assignment doesn't work out.

        Args:
            var: Variable to assign
            value: Value to assign to var

        Returns:
            List of (variable, value) pairs that were removed (for restoration)
        """
        self.support_pruning()
        # Record all values we're about to remove from var's domain
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        # Restrict var's domain to just this value
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Remove a value from a variable's current domain.

        This is the basic operation of constraint propagation: when we
        discover that var=value would lead to a contradiction, we eliminate
        that possibility.

        Args:
            var: Variable to prune
            value: Value to remove from var's domain
            removals: List to track removals (for backtracking), can be None
        """
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))  # Remember what we removed

    def choices(self, var):
        """Get current possible values for a variable.

        Returns curr_domains[var] if we're doing constraint propagation,
        otherwise returns the original domains[var].

        Args:
            var: Variable to get choices for

        Returns:
            List of currently possible values for var
        """
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        """Extract assignments that are forced by constraint propagation.

        After constraint propagation, some variables might have only one
        possible value left. This method finds those "forced" assignments.

        Returns:
            Dictionary of {var: value} for variables with only one choice left
        """
        self.support_pruning()
        return {
            v: self.curr_domains[v][0]  # Take the only remaining value
            for v in self.variables
            if 1 == len(self.curr_domains[v])  # Only one choice left
        }

    def restore(self, removals):
        """Undo constraint propagation changes when backtracking.

        When a search path doesn't work out, we need to restore the domains
        to their previous state before trying a different path.

        Args:
            removals: List of (var, value) pairs to restore
        """
        for B, b in removals:
            self.curr_domains[B].append(b)  # Add back the removed value

    # === Local Search Support ===

    def conflicted_vars(self, current):
        """Find variables that violate constraints in the current complete assignment.

        This is used by local search algorithms like min-conflicts that start
        with a complete (but possibly invalid) assignment and try to fix violations.

        Args:
            current: Complete assignment dictionary {var: value}

        Returns:
            List of variables that are currently in conflict with their neighbors
        """
        return [
            var
            for var in self.variables
            if self.nconflicts(var, current[var], current) > 0
        ]


# ______________________________________________________________________________
# ARC CONSISTENCY ALGORITHMS
#
# Arc consistency is a constraint propagation technique that eliminates values
# from variable domains when those values cannot possibly be part of a solution.
#
# Key concept: An arc (Xi, Xj) is consistent if for every value x in Xi's domain,
# there exists at least one value y in Xj's domain such that the constraint
# between Xi=x and Xj=y is satisfied.
#
# If an arc is not consistent, we can safely remove the "unsupported" values
# from Xi's domain. We repeat this process until all arcs are consistent or
# we discover the problem is unsolvable.


def no_arc_heuristic(csp, queue):
    """Simplest arc selection strategy: process arcs in whatever order they're given.

    This is the baseline heuristic that doesn't try to be smart about ordering.
    More sophisticated heuristics can lead to fewer constraint checks.

    Args:
        csp: The constraint satisfaction problem (unused in this simple heuristic)
        queue: Collection of arcs (Xi, Xj) to process

    Returns:
        The unmodified queue (no reordering)
    """
    return queue


def dom_j_up(csp, queue):
    """Smart arc selection: prioritize arcs where the second variable has fewer values.

    This heuristic is based on the idea that variables with smaller domains are
    more likely to become empty (leading to early failure detection) or become
    singleton (forcing an assignment). Processing these arcs first can lead to
    more efficient propagation.

    Strategy: Order arcs (Xi, Xj) by |domain(Xj)| in ascending order
             (smallest domains first)

    Args:
        csp: CSP with curr_domains containing current domain sizes
        queue: Collection of arcs (Xi, Xj) to be ordered

    Returns:
        SortedSet that automatically maintains arcs in order of increasing |domain(Xj)|
    """
    return SortedSet(queue, key=lambda t: neg(len(csp.curr_domains[t[1]])))


def AC3(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    """AC-3: The fundamental arc consistency algorithm [Figure 6.3 in textbook].

    AC-3 works by repeatedly checking and "revising" arcs until either:
    1. All arcs are consistent (success - problem may be solvable), or
    2. Some variable's domain becomes empty (failure - problem unsolvable)

    How it works:
    1. Start with a queue of all arcs in the CSP
    2. While queue is not empty:
       a. Remove an arc (Xi, Xj) from queue
       b. Check if Xi's domain needs revision due to Xj
       c. If revised and Xi's domain becomes empty → FAIL
       d. If revised → add arcs (Xk, Xi) back to queue for all neighbors Xk of Xi
    3. If we finish without failure → all arcs are consistent

    Think of it as: "If I change Xi's possible values, I might need to
    recheck how Xi affects its other neighbors."

    Args:
        csp: The constraint satisfaction problem
        queue: Set of arcs to check (default: all arcs in the CSP)
        removals: List to track pruned values (for backtracking), can be None
        arc_heuristic: Function to order arc processing for efficiency

    Returns:
        tuple: (is_consistent: bool, constraint_checks: int)
            - is_consistent: True if CSP remains consistent after propagation
            - constraint_checks: Number of times we checked constraints (performance metric)

    Time Complexity: O(cd³) where c=#constraints, d=max domain size
    Space Complexity: O(c) for the queue
    """
    if queue is None:
        # Create all possible arcs: for each variable Xi and each neighbor Xk
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()  # Initialize curr_domains if not already done
    queue = arc_heuristic(csp, queue)  # Apply ordering heuristic
    checks = 0  # Count constraint evaluations for performance analysis

    while queue:
        (Xi, Xj) = queue.pop()  # Get next arc to process
        revised, checks = revise(csp, Xi, Xj, removals, checks)
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # Xi has no values left → problem unsolvable
            # Xi changed, so we need to recheck how Xi affects its other neighbors
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:  # Don't recheck the arc we just processed
                    queue.add((Xk, Xi))
    return True, checks  # All arcs are consistent


def revise(csp, Xi, Xj, removals, checks=0):
    """Check and revise the domain of Xi to ensure arc (Xi, Xj) is consistent.

    This is the core operation of arc consistency. An arc (Xi, Xj) is consistent
    if every value in Xi's domain has at least one "supporting" value in Xj's domain
    (i.e., a value that satisfies the constraint between Xi and Xj).

    Algorithm:
    1. For each value x in Xi's current domain:
       2. Check if there exists any value y in Xj's domain such that
          constraints(Xi, x, Xj, y) is satisfied
       3. If no such y exists, remove x from Xi's domain (it's "unsupported")

    Example: Map coloring with Xi=WA, Xj=NT, constraint="different colors"
    - If WA can be {red, blue} and NT can be {red}, then WA=red has no support
    - We remove red from WA's domain, leaving WA={blue}

    Args:
        csp: The constraint satisfaction problem
        Xi: Variable whose domain we might revise (the "tail" of the arc)
        Xj: Variable that constrains Xi (the "head" of the arc)
        removals: List to track removed values (for backtracking), can be None
        checks: Running count of constraint evaluations

    Returns:
        tuple: (revised: bool, updated_checks: int)
            - revised: True if we removed any values from Xi's domain
            - updated_checks: New count including constraint checks made here
    """
    revised = False
    # Check each value in Xi's domain (use [:] to copy since we might modify the list)
    for x in csp.curr_domains[Xi][:]:
        # Look for a supporting value in Xj's domain
        conflict = True  # Assume x is unsupported until we find support
        for y in csp.curr_domains[Xj]:
            if csp.constraints(Xi, x, Xj, y):
                conflict = False  # Found support! x is OK
            checks += 1  # Count this constraint evaluation
            if not conflict:
                break  # No need to check more values - we found support

        # If no supporting value found, remove x from Xi's domain
        if conflict:
            csp.prune(Xi, x, removals)
            revised = True
    return revised, checks


# Constraint Propagation with AC3b: an improved version
# of AC3 with double-support domain-heuristic


def AC3b(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    """Improved AC-3 algorithm with double-support domain heuristic.

    An optimized version of AC3 that uses bidirectional support checking
    to reduce redundant constraint evaluations by maintaining sets of
    supported and unsupported values.

    Args:
        csp: The constraint satisfaction problem
        queue: Initial set of arcs to check, defaults to all arcs
        removals: List to track removed values for backtracking
        arc_heuristic: Function to order arc processing

    Returns:
        tuple: (is_consistent: bool, constraint_checks: int)
            - is_consistent: True if CSP remains consistent
            - constraint_checks: Number of constraint evaluations performed

    Time Complexity: Generally better than AC3 due to double-support optimization
    """
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        # Si_p values are all known to be supported by Xj
        # Sj_p values are all known to be supported by Xi
        # Dj - Sj_p = Sj_u values are unknown, as yet, to be supported by Xi
        Si_p, Sj_p, Sj_u, checks = partition(csp, Xi, Xj, checks)
        if not Si_p:
            return False, checks  # CSP is inconsistent
        revised = False
        for x in set(csp.curr_domains[Xi]) - Si_p:
            csp.prune(Xi, x, removals)
            revised = True
        if revised:
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
        if (Xj, Xi) in queue:
            if isinstance(queue, set):
                # Alternative removal methods: queue -= {(Xj, Xi)} or queue.remove((Xj, Xi))
                queue.difference_update({(Xj, Xi)})
            else:
                queue.difference_update((Xj, Xi))
            # the elements in D_j which are supported by Xi are given by the union of Sj_p with the set of those
            # elements of Sj_u which further processing will show to be supported by some vi_p in Si_p
            for vj_p in Sj_u:
                for vi_p in Si_p:
                    conflict = True
                    if csp.constraints(Xj, vj_p, Xi, vi_p):
                        conflict = False
                        Sj_p.add(vj_p)
                    checks += 1
                    if not conflict:
                        break
            revised = False
            for x in set(csp.curr_domains[Xj]) - Sj_p:
                csp.prune(Xj, x, removals)
                revised = True
            if revised:
                for Xk in csp.neighbors[Xj]:
                    if Xk != Xi:
                        queue.add((Xk, Xj))
    return True, checks  # CSP is satisfiable


def partition(csp, Xi, Xj, checks=0):
    """Partition domain values into supported and unsupported sets for AC3b.

    Implements the double-support optimization by categorizing values from
    both variable domains based on their mutual support relationships.

    Args:
        csp: The constraint satisfaction problem
        Xi: First variable in the arc
        Xj: Second variable in the arc
        checks: Current count of constraint checks

    Returns:
        tuple: (Si_p, Sj_p, Sj_u, checks)
            - Si_p: Set of Xi values supported by some value in Xj
            - Sj_p: Set of Xj values that support some value in Xi
            - Sj_u: Set of Xj values with unknown support status
            - checks: Updated count of constraint evaluations
    """
    Si_p = set()
    Sj_p = set()
    Sj_u = set(csp.curr_domains[Xj])
    for vi_u in csp.curr_domains[Xi]:
        conflict = True
        # now, in order to establish support for a value vi_u in Di it seems better to try to find a support among
        # the values in Sj_u first, because for each vj_u in Sj_u the check (vi_u, vj_u) is a double-support check
        # and it is just as likely that any vj_u in Sj_u supports vi_u than it is that any vj_p in Sj_p does...
        for vj_u in Sj_u - Sj_p:
            # double-support check
            if csp.constraints(Xi, vi_u, Xj, vj_u):
                conflict = False
                Si_p.add(vi_u)
                Sj_p.add(vj_u)
            checks += 1
            if not conflict:
                break
        # ... and only if no support can be found among the elements in Sj_u, should the elements vj_p in Sj_p be used
        # for single-support checks (vi_u, vj_p)
        if conflict:
            for vj_p in Sj_p:
                # single-support check
                if csp.constraints(Xi, vi_u, Xj, vj_p):
                    conflict = False
                    Si_p.add(vi_u)
                checks += 1
                if not conflict:
                    break
    return Si_p, Sj_p, Sj_u - Sj_p, checks


# Constraint Propagation with AC4


def AC4(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    """AC-4 arc consistency algorithm with support counting.

    Alternative to AC-3 that maintains explicit support counters for each
    value-variable pair, enabling more efficient propagation of constraint
    violations through the network.

    Args:
        csp: The constraint satisfaction problem
        queue: Initial set of arcs to check, defaults to all arcs
        removals: List to track removed values for backtracking
        arc_heuristic: Function to order arc processing

    Returns:
        tuple: (is_consistent: bool, constraint_checks: int)
            - is_consistent: True if CSP remains consistent
            - constraint_checks: Number of constraint evaluations performed

    Time Complexity: O(cd^2) preprocessing + O(cd) propagation per removal
    Space Complexity: O(cd^2) for support data structures
    """
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    support_counter = Counter()
    variable_value_pairs_supported = defaultdict(set)
    unsupported_variable_value_pairs = []
    checks = 0
    # construction and initialization of support sets
    while queue:
        (Xi, Xj) = queue.pop()
        revised = False
        for x in csp.curr_domains[Xi][:]:
            for y in csp.curr_domains[Xj]:
                if csp.constraints(Xi, x, Xj, y):
                    support_counter[(Xi, x, Xj)] += 1
                    variable_value_pairs_supported[(Xj, y)].add((Xi, x))
                checks += 1
            if support_counter[(Xi, x, Xj)] == 0:
                csp.prune(Xi, x, removals)
                revised = True
                unsupported_variable_value_pairs.append((Xi, x))
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
    # propagation of removed values
    while unsupported_variable_value_pairs:
        Xj, y = unsupported_variable_value_pairs.pop()
        for Xi, x in variable_value_pairs_supported[(Xj, y)]:
            revised = False
            if x in csp.curr_domains[Xi][:]:
                support_counter[(Xi, x, Xj)] -= 1
                if support_counter[(Xi, x, Xj)] == 0:
                    csp.prune(Xi, x, removals)
                    revised = True
                    unsupported_variable_value_pairs.append((Xi, x))
            if revised:
                if not csp.curr_domains[Xi]:
                    return False, checks  # CSP is inconsistent
    return True, checks  # CSP is satisfiable


# ______________________________________________________________________________
# BACKTRACKING SEARCH FOR CSPs
#
# Backtracking is the most common method for solving CSPs exactly. It works by:
# 1. Choose an unassigned variable
# 2. Try assigning it each possible value
# 3. If a value leads to conflicts, backtrack and try the next value
# 4. Use heuristics and constraint propagation to make this more efficient
#
# The basic backtracking algorithm can be enhanced with:
# - Variable ordering heuristics (which variable to assign next?)
# - Value ordering heuristics (which value to try first for a variable?)
# - Inference/constraint propagation (eliminate impossible values early)

# === Variable Ordering Heuristics ===
# These help decide which variable to assign next


def first_unassigned_variable(assignment, csp):
    """Simplest variable selection: just pick the first unassigned variable.

    This is the baseline strategy with no intelligence. Usually not the best choice,
    but simple and always works.

    Args:
        assignment: Current partial assignment
        csp: The constraint satisfaction problem

    Returns:
        First variable that doesn't have a value in assignment, or None if all assigned
    """
    return first([var for var in csp.variables if var not in assignment])


def mrv(assignment, csp):
    """Most Remaining Values (MRV) heuristic: choose variable with fewest legal values.

    Also called the "fail-first" heuristic. The idea is that variables with fewer
    remaining values are more likely to fail soon, so we should try them first
    to detect failures early (and avoid wasted work on dead-end paths).

    Example: If variable A has 2 possible values and variable B has 5,
             choose A first since it's more constrained.

    Args:
        assignment: Current partial assignment
        csp: The constraint satisfaction problem

    Returns:
        Unassigned variable with smallest domain (ties broken randomly)
    """
    return argmin_random_tie(
        [v for v in csp.variables if v not in assignment],
        key=lambda var: num_legal_values(csp, var, assignment),
    )


def num_legal_values(csp, var, assignment):
    """Count how many values are currently legal for a variable.

    This is a helper function used by the MRV heuristic. A value is "legal"
    if assigning it to the variable wouldn't immediately violate any constraints
    with already-assigned variables.

    Args:
        csp: The constraint satisfaction problem
        var: Variable to count legal values for
        assignment: Current partial assignment

    Returns:
        int: Number of values in var's domain that don't cause immediate conflicts
    """
    if csp.curr_domains:
        # If we're doing constraint propagation, curr_domains already has legal values
        return len(csp.curr_domains[var])
    else:
        # Otherwise, count values that don't conflict with current assignment
        return count(
            csp.nconflicts(var, val, assignment) == 0 for val in csp.domains[var]
        )


# === Value Ordering Heuristics ===
# These help decide which value to try first for a chosen variable


def unordered_domain_values(var, assignment, csp):
    """Simplest value ordering: try values in whatever order they're stored.

    This is the baseline with no intelligence about which values are more
    promising to try first.

    Args:
        var: Variable we're about to assign
        assignment: Current partial assignment (unused in this simple heuristic)
        csp: The constraint satisfaction problem

    Returns:
        List of possible values for var in their original order
    """
    return csp.choices(var)


def lcv(var, assignment, csp):
    """Least Constraining Value (LCV) heuristic: try values that rule out fewest choices.

    The idea is to try values that keep maximum flexibility for future assignments.
    A value is "more constraining" if assigning it would eliminate many possibilities
    for other unassigned variables.

    Example: In map coloring, if assigning "red" to a region would force 3 neighbors
             to avoid red, but assigning "blue" would only constrain 1 neighbor,
             try blue first.

    Args:
        var: Variable we're about to assign
        assignment: Current partial assignment
        csp: The constraint satisfaction problem

    Returns:
        Values for var sorted by how many conflicts they'd cause (least first)
    """
    return sorted(
        csp.choices(var), key=lambda val: csp.nconflicts(var, val, assignment)
    )


# === Inference/Constraint Propagation for Backtracking ===
# These functions run after each assignment to eliminate impossible values
# from remaining variables, making the search more efficient


def no_inference(csp, var, value, assignment, removals):
    """No constraint propagation: just accept the assignment as-is.

    This is the simplest approach - make assignments and only detect conflicts
    when we actually try to assign conflicting values. No "look ahead."

    Args:
        csp: The constraint satisfaction problem (unused)
        var: Variable that was just assigned (unused)
        value: Value that was assigned (unused)
        assignment: Current assignment (unused)
        removals: List to track removed values (unused)

    Returns:
        bool: Always True (never detects inconsistency early)
    """
    return True


def forward_checking(csp, var, value, assignment, removals):
    """Forward checking: eliminate values from neighbors that conflict with new assignment.

    When we assign var=value, look at each unassigned neighbor and remove any
    values from their domains that would violate constraints with var=value.
    This catches some inconsistencies early without full arc consistency.

    Example: In map coloring, if we assign WA=red, immediately remove red
             from domains of NT and SA (WA's neighbors).

    Args:
        csp: The constraint satisfaction problem
        var: Variable that was just assigned
        value: Value that was assigned to var
        assignment: Current partial assignment
        removals: List to track removed values (for backtracking)

    Returns:
        bool: False if any neighbor's domain becomes empty, True otherwise
    """
    csp.support_pruning()
    for B in csp.neighbors[var]:
        if B not in assignment:  # Only check unassigned neighbors
            # Remove values from B's domain that conflict with var=value
            for b in csp.curr_domains[B][:]:  # Copy list since we'll modify it
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            # If B has no values left, we've detected inconsistency
            if not csp.curr_domains[B]:
                return False
    return True


def mac(csp, var, value, assignment, removals, constraint_propagation=AC3b):
    """Maintaining Arc Consistency (MAC): run full arc consistency after each assignment.

    This is the most powerful inference method. After assigning var=value,
    run arc consistency on all arcs involving var to propagate the effects
    as far as possible through the constraint network.

    MAC often dramatically reduces the search space but costs more per node.

    Args:
        csp: The constraint satisfaction problem
        var: Variable that was just assigned
        value: Value assigned to var
        assignment: Current partial assignment (unused here)
        removals: List to track removed values for backtracking
        constraint_propagation: Which AC algorithm to use (default: AC3b)

    Returns:
        bool: False if arc consistency detects inconsistency, True otherwise
    """
    # Run arc consistency on all arcs pointing toward the newly assigned variable
    result, _ = constraint_propagation(
        csp, {(X, var) for X in csp.neighbors[var]}, removals
    )
    return result


# === The Main Backtracking Search Algorithm ===


def backtracking_search(
    csp,
    select_unassigned_variable=first_unassigned_variable,
    order_domain_values=unordered_domain_values,
    inference=no_inference,
):
    """Backtracking search algorithm for solving CSPs [Figure 6.5 in textbook].

    This is the workhorse algorithm for exact CSP solving. It systematically
    explores the space of partial assignments, backtracking when it hits
    contradictions.

    Basic algorithm:
    1. If all variables assigned → return solution
    2. Choose an unassigned variable (using heuristic)
    3. For each value in variable's domain (using ordering heuristic):
       a. If value doesn't conflict, assign it
       b. Apply inference/constraint propagation
       c. Recursively solve remaining problem
       d. If recursive call succeeds → return solution
       e. Otherwise, undo assignment and inference (backtrack)
    4. If no value works → return failure

    The three heuristic functions allow you to customize the search strategy:
    - Variable selection can use MRV, degree heuristic, etc.
    - Value ordering can use LCV, random, etc.
    - Inference can be none, forward checking, MAC, etc.

    Args:
        csp: The constraint satisfaction problem to solve
        select_unassigned_variable: Function(assignment, csp) → variable
        order_domain_values: Function(var, assignment, csp) → ordered values
        inference: Function(csp, var, value, assignment, removals) → bool

    Returns:
        dict: Complete valid assignment if solution found
        None: If no solution exists

    Time Complexity: O(d^n) worst case, but heuristics and inference can help dramatically
    Space Complexity: O(n) for recursion depth
    """

    def backtrack(assignment):
        """Recursive helper function that does the actual backtracking."""
        # BASE CASE: If all variables are assigned, we have a solution
        if len(assignment) == len(csp.variables):
            return assignment

        # RECURSIVE CASE: Choose a variable and try values for it
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            # Only try values that don't immediately conflict
            if 0 == csp.nconflicts(var, value, assignment):
                # Make the assignment
                csp.assign(var, value, assignment)
                # Record what inference removes (so we can undo it)
                removals = csp.suppose(var, value)
                # Apply constraint propagation/inference
                if inference(csp, var, value, assignment, removals):
                    # Inference didn't detect contradiction, so recurse
                    result = backtrack(assignment)
                    if result is not None:
                        return result  # Found solution in recursive call!
                # If we get here, this assignment didn't work out
                csp.restore(removals)  # Undo inference changes

        # If we get here, no value worked for this variable
        csp.unassign(var, assignment)  # Undo the assignment
        return None  # Signal failure to parent call

    # Start the search with an empty assignment
    result = backtrack({})
    # Sanity check: if we claim to have a solution, verify it's actually valid
    assert result is None or csp.goal_test(result)
    return result


# ______________________________________________________________________________
# LOCAL SEARCH: MIN-CONFLICTS ALGORITHM
#
# Sometimes backtracking is too slow (especially for large problems).
# Local search offers an alternative approach that's often much faster:
#
# 1. Start with a complete assignment (even if it violates some constraints)
# 2. Repeatedly fix violations by changing one variable at a time
# 3. Stop when no violations remain (success) or time limit reached (failure)
#
# This works surprisingly well for many large CSPs where systematic search fails.


def min_conflicts(csp, max_steps=100000):
    """Solve CSP using min-conflicts local search algorithm.

    Local search strategy that starts with a complete assignment and iteratively
    improves it by resolving conflicts. This is often much faster than backtracking
    for large problems, especially those with loose constraints.

    Algorithm:
    1. Generate initial complete assignment (probably with conflicts)
    2. Repeat until solution found or max_steps reached:
       a. If no conflicts remain → return solution
       b. Pick a random variable that's currently in conflict
       c. Change it to the value that minimizes total conflicts
       d. Continue...

    Why this works: Many CSPs have the property that local minima are rare
    and you can usually "hill climb" to a solution.

    Args:
        csp: The constraint satisfaction problem to solve
        max_steps: Maximum number of steps before giving up

    Returns:
        dict: Complete assignment if solution found
        None: If no solution found within max_steps

    Time Complexity: O(max_steps) but each step is usually fast
    Space Complexity: O(n) for the assignment
    Best for: Large, loosely constrained problems where backtracking is too slow
    """
    # Step 1: Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)

    # Step 2: Iteratively fix conflicts by reassigning conflicted variables
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current  # Success! No conflicts remain

        # Pick a random conflicted variable and reassign it to minimize conflicts
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)

    return None  # Failure: couldn't solve within max_steps


def min_conflicts_value(csp, var, current):
    """Find the value for var that minimizes conflicts in the current assignment.

    This is the key heuristic for min-conflicts search. Given a variable,
    we want to assign it the value that causes the fewest constraint violations
    with other currently assigned variables.

    Example: In N-Queens, if we're reassigning the queen in column 3,
             try each possible row and count how many other queens it would attack.
             Choose the row with the minimum number of attacks.

    Args:
        csp: The constraint satisfaction problem
        var: Variable to find the best value for
        current: Current complete assignment

    Returns:
        Value from var's domain that minimizes conflicts (ties broken randomly)
    """
    return argmin_random_tie(
        csp.domains[var], key=lambda val: csp.nconflicts(var, val, current)
    )


# ______________________________________________________________________________


def tree_csp_solver(csp):
    """Solve tree-structured CSPs in linear time [Figure 6.11].

    Specialized algorithm for CSPs whose constraint graph forms a tree.
    Uses topological ordering and arc consistency to solve efficiently.

    Args:
        csp: Tree-structured constraint satisfaction problem

    Returns:
        dict or None: Complete assignment if solvable, None if inconsistent

    Time Complexity: O(nd) where n=variables, d=domain size
    Precondition: CSP constraint graph must be a tree
    """
    assignment = {}
    root = csp.variables[0]
    X, parent = topological_sort(csp, root)

    csp.support_pruning()
    for Xj in reversed(X[1:]):
        if not make_arc_consistent(parent[Xj], Xj, csp):
            return None

    assignment[root] = csp.curr_domains[root][0]
    for Xi in X[1:]:
        assignment[Xi] = assign_value(parent[Xi], Xi, csp, assignment)
        if not assignment[Xi]:
            return None
    return assignment


def topological_sort(X, root):
    """Returns the topological sort of X starting from the root.

    Performs depth-first traversal to establish parent-child relationships
    and creates a topological ordering suitable for tree CSP solving.

    Args:
        X: CSP object with variables and neighbor relationships
        root: Starting variable for the topological sort

    Returns:
        tuple: (ordered_variables, parent_dict)
            - ordered_variables: List of variables in topological order
            - parent_dict: Dict mapping each variable to its parent (None for root)

    Note:
        Used internally by tree_csp_solver for efficient tree processing.
    """
    neighbors = X.neighbors

    visited = defaultdict(lambda: False)

    stack = []
    parents = {}

    build_topological(root, None, neighbors, visited, stack, parents)
    return stack, parents


def build_topological(node, parent, neighbors, visited, stack, parents):
    """Build the topological sort and parent relationships through DFS.

    Recursive helper function that performs depth-first traversal to
    establish the topological ordering and parent-child relationships
    needed for tree CSP solving.

    Args:
        node: Current node being processed
        parent: Parent of current node (None for root)
        neighbors: Dict mapping nodes to their neighbors
        visited: Dict tracking visited status of nodes
        stack: List being built with topological ordering
        parents: Dict being built with parent relationships
    """
    visited[node] = True

    for n in neighbors[node]:
        if not visited[n]:
            build_topological(n, node, neighbors, visited, stack, parents)

    parents[node] = parent
    stack.insert(0, node)


def make_arc_consistent(Xj, Xk, csp):
    """Make arc between parent (Xj) and child (Xk) consistent under the csp's constraints.

    Removes values from parent variable Xj's domain that have no supporting
    value in child variable Xk's domain. Used in tree CSP preprocessing.

    Args:
        Xj: Parent variable in the tree
        Xk: Child variable in the tree
        csp: The constraint satisfaction problem

    Returns:
        list: Remaining domain values for Xj after pruning

    Side Effects:
        Modifies csp.curr_domains[Xj] by removing inconsistent values
    """
    for val1 in csp.domains[Xj]:
        keep = False  # Keep or remove val1
        for val2 in csp.domains[Xk]:
            if csp.constraints(Xj, val1, Xk, val2):
                # Found a consistent assignment for val1, keep it
                keep = True
                break

        if not keep:
            # Remove val1
            csp.prune(Xj, val1, None)

    return csp.curr_domains[Xj]


def assign_value(Xj, Xk, csp, assignment):
    """Assign a value to Xk given Xj's (Xk's parent) assignment.

    For tree CSPs, finds the first value in Xk's domain that is
    consistent with the parent variable Xj's assigned value.

    Args:
        Xj: Parent variable (already assigned)
        Xk: Child variable to assign
        csp: The constraint satisfaction problem
        assignment: Current partial assignment including Xj

    Returns:
        Value for Xk that satisfies constraints with Xj, or None if none exists
    """
    parent_assignment = assignment[Xj]
    for val in csp.curr_domains[Xk]:
        if csp.constraints(Xj, parent_assignment, Xk, val):
            return val

    # No consistent assignment available
    return None


# ______________________________________________________________________________
# CLASSIC CSP PROBLEMS
#
# This section contains implementations of famous CSP problems that are
# commonly used for teaching, benchmarking, and testing CSP algorithms.


class UniversalDict:
    """A dictionary that returns the same value for any key.

    This is a convenient data structure for CSPs where all variables have
    the same domain. Instead of creating a large dictionary like:
    {'A': [1,2,3], 'B': [1,2,3], 'C': [1,2,3], ...}

    We can just use:
    UniversalDict([1,2,3])

    Example usage:
        >>> d = UniversalDict(42)
        >>> d['anything']
        42
        >>> d[123]
        42
    """

    def __init__(self, value):
        """Create a universal dictionary that maps everything to value."""
        self.value = value

    def __getitem__(self, key):
        """Return the universal value regardless of what key is requested."""
        return self.value

    def __repr__(self):
        """String representation showing this maps anything to the value."""
        return "{{Any: {0!r}}}".format(self.value)


def different_values_constraint(A, a, B, b):
    """Basic inequality constraint: neighboring variables must have different values.

    This is the most common constraint in CSPs like map coloring, graph coloring,
    and scheduling problems where adjacent/related items cannot be the same.

    Args:
        A: First variable name (e.g., 'WA' for Western Australia)
        a: Value for first variable (e.g., 'red')
        B: Second variable name (e.g., 'NT' for Northern Territory)
        b: Value for second variable (e.g., 'blue')

    Returns:
        bool: True if the values are different (constraint satisfied)
              False if the values are the same (constraint violated)

    Example:
        >>> different_values_constraint('WA', 'red', 'NT', 'blue')
        True  # Adjacent regions can have different colors
        >>> different_values_constraint('WA', 'red', 'NT', 'red')
        False # Adjacent regions cannot have the same color
    """
    return a != b


def MapColoringCSP(colors, neighbors):
    """Create a CSP for the classic map coloring problem.

    Map coloring is one of the most famous CSP problems: given a map with regions
    and a set of colors, assign colors to regions so that no two adjacent regions
    have the same color.

    This problem is important because:
    - It's easy to understand but can be computationally hard
    - It demonstrates key CSP concepts clearly
    - It has real applications (e.g., frequency assignment, register allocation)
    - The 4-color theorem proves that any planar map can be colored with 4 colors

    Args:
        colors: List/sequence of available colors (e.g., ['red', 'green', 'blue'])
        neighbors: Region adjacency information, either:
                  - Dict: {'region': [adjacent_regions]}
                  - String: 'X: Y Z; Y: Z' format (see parse_neighbors)

    Returns:
        CSP: Configured constraint satisfaction problem for map coloring

    Example:
        # Simple 3-region map
        >>> csp = MapColoringCSP(['red','blue'],
        ...                     {'A': ['B'], 'B': ['A','C'], 'C': ['B']})
        # String format is more convenient for larger maps
        >>> australia = MapColoringCSP(['R','G','B'],
        ...                           'WA: NT SA; NT: WA SA Q; SA: WA NT Q NSW V')
    """
    if isinstance(neighbors, str):
        neighbors = parse_neighbors(neighbors)  # Convert string to dict format

    return CSP(
        list(neighbors.keys()),  # Variables: all regions
        UniversalDict(colors),  # Domains: same colors for all regions
        neighbors,  # Neighbors: adjacency relationships
        different_values_constraint,  # Constraints: adjacent regions must differ
    )


def parse_neighbors(neighbors):
    """Convert a string specification into a neighbors dictionary.

    This utility function lets you specify map adjacencies in a compact string format
    instead of typing out a full dictionary. This is much more convenient for
    larger maps.

    String format rules:
    - Each region specification: 'Region: neighbor1 neighbor2 ...'
    - Separate region specs with ';'
    - Adjacency is symmetric: if you list 'A: B', you don't need 'B: A'
    - Whitespace is ignored

    Args:
        neighbors: String like 'X: Y Z; Y: Z W' meaning:
                  - X is adjacent to Y and Z
                  - Y is adjacent to Z and W
                  - (symmetric relationships are added automatically)

    Returns:
        dict: {region: [list_of_adjacent_regions]} with symmetric entries

    Example:
        >>> parse_neighbors('X: Y Z; Y: Z')
        {'X': ['Y', 'Z'], 'Y': ['X', 'Z'], 'Z': ['X', 'Y']}

        This means X neighbors Y&Z, Y neighbors X&Z, Z neighbors X&Y
    """
    dic = defaultdict(list)  # Automatically creates empty lists for new keys

    # Split into individual region specifications
    specs = [spec.split(":") for spec in neighbors.split(";")]

    for A, Aneighbors in specs:
        A = A.strip()  # Remove whitespace
        # Add each neighbor relationship (and its reverse)
        for B in Aneighbors.split():
            dic[A].append(B)  # A neighbors B
            dic[B].append(A)  # B neighbors A (symmetric)
    return dic


# === Example Map Coloring Problems ===
# These are classic examples used in AI textbooks and courses

# Australia map coloring - a simple example with 7 regions
australia_csp = MapColoringCSP(
    list("RGB"),  # 3 colors: Red, Green, Blue
    """SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: """,
    # SA (South Australia) borders: WA, NT, Q, NSW, V
    # NT (Northern Territory) borders: WA, Q
    # NSW (New South Wales) borders: Q, V
    # T (Tasmania) is an island - no borders
)

# USA map coloring - all 50 states (more challenging)
usa_csp = MapColoringCSP(
    list("RGBY"),  # 4 colors needed for USA
    """WA: OR ID; OR: ID NV CA; CA: NV AZ; NV: ID UT AZ; ID: MT WY UT;
                         UT: WY CO AZ; MT: ND SD WY; WY: SD NE CO; CO: NE KA OK NM; NM: OK TX AZ;
                         ND: MN SD; SD: MN IA NE; NE: IA MO KA; KA: MO OK; OK: MO AR TX;
                         TX: AR LA; MN: WI IA; IA: WI IL MO; MO: IL KY TN AR; AR: MS TN LA;
                         LA: MS; WI: MI IL; IL: IN KY; IN: OH KY; MS: TN AL; AL: TN GA FL;
                         MI: OH IN; OH: PA WV KY; KY: WV VA TN; TN: VA NC GA; GA: NC SC FL;
                         PA: NY NJ DE MD WV; WV: MD VA; VA: MD DC NC; NC: SC; NY: VT MA CT NJ;
                         NJ: DE; DE: MD; MD: DC; VT: NH MA; MA: NH RI CT; CT: RI; ME: NH;
                         HI: ; AK: """,
    # Note: HI (Hawaii) and AK (Alaska) have no borders with other states
)

# France map coloring - European example
france_csp = MapColoringCSP(
    list("RGBY"),  # 4 colors for French regions
    """AL: LO FC; AQ: MP LI PC; AU: LI CE BO RA LR MP; BO: CE IF CA FC RA
                            AU; BR: NB PL; CA: IF PI LO FC BO; CE: PL NB NH IF BO AU LI PC; FC: BO
                            CA LO AL RA; IF: NH PI CA BO CE; LI: PC CE AU MP AQ; LO: CA AL FC; LR:
                            MP AU RA PA; MP: AQ LI AU LR; NB: NH CE PL BR; NH: PI IF CE NB; NO:
                            PI; PA: LR RA; PC: PL CE LI AQ; PI: NH NO CA IF; PL: BR NB CE PC; RA:
                            AU BO FC PA LR""",
)


# ______________________________________________________________________________
# N-QUEENS PROBLEM
#
# The N-Queens problem: place N queens on an N×N chessboard so that no two
# queens attack each other. This is a classic CSP that demonstrates:
# - How to model problems with implicit constraints
# - The power of good variable/constraint representations
# - Specialized data structures for efficiency
#
# Our representation:
# - Variables: columns (0, 1, 2, ..., N-1)
# - Values: rows (0, 1, 2, ..., N-1)
# - Constraints: no two queens in same row, diagonal, or anti-diagonal
# - Assignment[col] = row means "place queen in column col, row row"


def queen_constraint(A, a, B, b):
    """Check if two queens can coexist on a chessboard.

    Queens attack each other if they're in the same:
    1. Row (same a and b)
    2. Diagonal (|A-B| = |a-b|, going up-right or down-right)
    3. Anti-diagonal (A+a = B+b, going up-left or down-left)

    Args:
        A: Column of first queen (0-indexed)
        a: Row of first queen (0-indexed)
        B: Column of second queen (0-indexed)
        b: Row of second queen (0-indexed)

    Returns:
        bool: True if queens don't attack each other (constraint satisfied)
              False if queens attack each other (constraint violated)

    Note: Position (A,a) means queen at column A, row a
    """
    return A == B or (a != b and A + a != B + b and A - a != B - b)
    # A == B handles the case where we're checking a variable against itself
    # a != b: different rows
    # A + a != B + b: not on same anti-diagonal
    # A - a != B - b: not on same diagonal


class NQueensCSP(CSP):
    """Optimized CSP implementation for the N-Queens problem.

    This specialized version uses efficient data structures to track conflicts,
    making it suitable for large N (hundreds or thousands of queens). The key
    optimization: instead of checking all pairs of queens for conflicts, we
    maintain counts of how many queens are in each row/diagonal.

    Data structures:
    - rows[i]: Number of queens in row i
    - downs[i]: Number of queens in diagonal i (where column + row = i)
    - ups[i]: Number of queens in anti-diagonal i (where column - row + N - 1 = i)

    Why this works: If rows[3] = 0, then row 3 is empty. If rows[3] = 1,
    then row 3 has exactly one queen. If rows[3] > 1, we have a conflict.

    Time complexities:
    - Moving a queen: O(1) (just update counters)
    - Counting conflicts for a position: O(1) (just check counters)
    - Choosing best move: O(N) (check all positions in a column)

    This efficiency makes min-conflicts search very fast for N-Queens.

    Example usage:
        >>> problem = NQueensCSP(8)  # 8x8 chessboard
        >>> solution = min_conflicts(problem)  # Usually solves quickly
        >>> len(solution) == 8  # Should find all 8 queen positions
        True
    """

    def __init__(self, n):
        """Create an N-Queens CSP for an n×n chessboard.

        Args:
            n: Size of chessboard (number of queens to place)
        """
        # Set up the basic CSP structure
        CSP.__init__(
            self,
            list(range(n)),  # Variables: columns 0,1,2,...,n-1
            UniversalDict(
                list(range(n))
            ),  # Domains: all variables can use rows 0,1,2,...,n-1
            UniversalDict(
                list(range(n))
            ),  # Neighbors: every column constrains every other column
            queen_constraint,  # Constraints: queens cannot attack each other
        )

        # Initialize conflict-tracking data structures
        self.rows = [0] * n  # rows[i] = number of queens in row i
        self.ups = [0] * (2 * n - 1)  # ups[i] = number of queens in anti-diagonal i
        self.downs = [0] * (2 * n - 1)  # downs[i] = number of queens in diagonal i

    def nconflicts(self, var, val, assignment):
        """The number of conflicts, as recorded with each assignment.
        Count conflicts in row and in up, down diagonals. If there
        is a queen there, it can't conflict with itself, so subtract 3."""
        n = len(self.variables)
        c = self.rows[val] + self.downs[var + val] + self.ups[var - val + n - 1]
        if assignment.get(var, None) == val:
            c -= 3
        return c

    def assign(self, var, val, assignment):
        """Assign var, and keep track of conflicts."""
        old_val = assignment.get(var, None)
        if val != old_val:
            if old_val is not None:  # Remove old val if there was one
                self.record_conflict(assignment, var, old_val, -1)
            self.record_conflict(assignment, var, val, +1)
            CSP.assign(self, var, val, assignment)

    def unassign(self, var, assignment):
        """Remove var from assignment (if it is there) and track conflicts."""
        if var in assignment:
            self.record_conflict(assignment, var, assignment[var], -1)
        CSP.unassign(self, var, assignment)

    def record_conflict(self, assignment, var, val, delta):
        """Record conflicts caused by addition or deletion of a Queen."""
        n = len(self.variables)
        self.rows[val] += delta
        self.downs[var + val] += delta
        self.ups[var - val + n - 1] += delta

    def display(self, assignment):
        """Print the queens and the nconflicts values (for debugging)."""
        n = len(self.variables)
        for val in range(n):
            for var in range(n):
                if assignment.get(var, "") == val:
                    ch = "Q"
                elif (var + val) % 2 == 0:
                    ch = "."
                else:
                    ch = "-"
                print(ch, end=" ")
            print("    ", end=" ")
            for var in range(n):
                if assignment.get(var, "") == val:
                    ch = "*"
                else:
                    ch = " "
                print(str(self.nconflicts(var, val, assignment)) + ch, end=" ")
            print()


# ______________________________________________________________________________
# Sudoku


def flatten(seqs):
    """Flatten a sequence of sequences into a single list.

    This utility function is commonly used when working with nested data structures
    like grids or matrices in puzzle problems.

    Args:
        seqs: Sequence of sequences (e.g., list of lists, list of tuples)

    Returns:
        list: Single flattened list containing all elements in order

    Example:
        >>> flatten([[1,2], [3,4], [5]])
        [1, 2, 3, 4, 5]
        >>> flatten([['a','b'], ['c'], ['d','e','f']])
        ['a', 'b', 'c', 'd', 'e', 'f']
    """
    return sum(seqs, [])  # Clever trick: sum with empty list as start value


easy1 = (
    "..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3.."
)
harder1 = (
    "4173698.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"
)

_R3 = list(range(3))
_CELL = itertools.count().__next__
_BGRID = [[[[_CELL() for x in _R3] for y in _R3] for bx in _R3] for by in _R3]
_BOXES = flatten([list(map(flatten, brow)) for brow in _BGRID])
_ROWS = flatten([list(map(flatten, zip(*brow))) for brow in _BGRID])
_COLS = list(zip(*_ROWS))

_NEIGHBORS = {v: set() for v in flatten(_ROWS)}
for unit in map(set, _BOXES + _ROWS + _COLS):
    for v in unit:
        _NEIGHBORS[v].update(unit - {v})


class Sudoku(CSP):
    """
    A Sudoku problem.
    The box grid is a 3x3 array of boxes, each a 3x3 array of cells.
    Each cell holds a digit in 1..9. In each box, all digits are
    different; the same for each row and column as a 9x9 grid.
    >>> e = Sudoku(easy1)
    >>> e.display(e.infer_assignment())
    . . 3 | . 2 . | 6 . .
    9 . . | 3 . 5 | . . 1
    . . 1 | 8 . 6 | 4 . .
    ------+-------+------
    . . 8 | 1 . 2 | 9 . .
    7 . . | . . . | . . 8
    . . 6 | 7 . 8 | 2 . .
    ------+-------+------
    . . 2 | 6 . 9 | 5 . .
    8 . . | 2 . 3 | . . 9
    . . 5 | . 1 . | 3 . .
    >>> AC3(e)  # doctest: +ELLIPSIS
    (True, ...)
    >>> e.display(e.infer_assignment())
    4 8 3 | 9 2 1 | 6 5 7
    9 6 7 | 3 4 5 | 8 2 1
    2 5 1 | 8 7 6 | 4 9 3
    ------+-------+------
    5 4 8 | 1 3 2 | 9 7 6
    7 2 9 | 5 6 4 | 1 3 8
    1 3 6 | 7 9 8 | 2 4 5
    ------+-------+------
    3 7 2 | 6 8 9 | 5 1 4
    8 1 4 | 2 5 3 | 7 6 9
    6 9 5 | 4 1 7 | 3 8 2
    >>> h = Sudoku(harder1)
    >>> backtracking_search(h, select_unassigned_variable=mrv, inference=forward_checking) is not None
    True
    """

    R3 = _R3
    Cell = _CELL
    bgrid = _BGRID
    boxes = _BOXES
    rows = _ROWS
    cols = _COLS
    neighbors = _NEIGHBORS

    def __init__(self, grid):
        """Build a Sudoku problem from a string representing the grid:
        the digits 1-9 denote a filled cell, '.' or '0' an empty one;
        other characters are ignored."""
        squares = iter(re.findall(r"\d|\.", grid))
        domains = {
            var: [ch] if ch in "123456789" else "123456789"
            for var, ch in zip(flatten(self.rows), squares)
        }
        for _ in squares:
            raise ValueError("Not a Sudoku grid", grid)  # Too many squares
        CSP.__init__(self, None, domains, self.neighbors, different_values_constraint)

    def display(self, assignment):
        def show_box(box):
            return [" ".join(map(show_cell, row)) for row in box]

        def show_cell(cell):
            return str(assignment.get(cell, "."))

        def abut(lines1, lines2):
            return list(map(" | ".join, list(zip(lines1, lines2))))

        print(
            "\n------+-------+------\n".join(
                "\n".join(reduce(abut, map(show_box, brow))) for brow in self.bgrid
            )
        )


# ______________________________________________________________________________
# The Zebra Puzzle


def Zebra():
    """Return an instance of the famous Zebra Puzzle CSP.

    The Zebra Puzzle (also known as Einstein's Riddle) is a logic puzzle
    involving five houses with different attributes. The goal is to determine
    who owns the zebra and who drinks water.

    Returns:
        CSP: Configured constraint satisfaction problem for the Zebra Puzzle

    Note: This is a classic CSP benchmark with exactly one solution.
    """
    Colors = "Red Yellow Blue Green Ivory".split()
    Pets = "Dog Fox Snails Horse Zebra".split()
    Drinks = "OJ Tea Coffee Milk Water".split()
    Countries = "Englishman Spaniard Norwegian Ukranian Japanese".split()
    Smokes = "Kools Chesterfields Winston LuckyStrike Parliaments".split()
    variables = Colors + Pets + Drinks + Countries + Smokes
    domains = {}
    for var in variables:
        domains[var] = list(range(1, 6))
    domains["Norwegian"] = [1]
    domains["Milk"] = [3]
    neighbors = parse_neighbors("""Englishman: Red;
                Spaniard: Dog; Kools: Yellow; Chesterfields: Fox;
                Norwegian: Blue; Winston: Snails; LuckyStrike: OJ;
                Ukranian: Tea; Japanese: Parliaments; Kools: Horse;
                Coffee: Green; Green: Ivory""")
    for type in [Colors, Pets, Drinks, Countries, Smokes]:
        for A in type:
            for B in type:
                if A != B:
                    if B not in neighbors[A]:
                        neighbors[A].append(B)
                    if A not in neighbors[B]:
                        neighbors[B].append(A)

    def zebra_constraint(A, a, B, b, recurse=0):
        same = a == b
        next_to = abs(a - b) == 1
        if A == "Englishman" and B == "Red":
            return same
        if A == "Spaniard" and B == "Dog":
            return same
        if A == "Chesterfields" and B == "Fox":
            return next_to
        if A == "Norwegian" and B == "Blue":
            return next_to
        if A == "Kools" and B == "Yellow":
            return same
        if A == "Winston" and B == "Snails":
            return same
        if A == "LuckyStrike" and B == "OJ":
            return same
        if A == "Ukranian" and B == "Tea":
            return same
        if A == "Japanese" and B == "Parliaments":
            return same
        if A == "Kools" and B == "Horse":
            return next_to
        if A == "Coffee" and B == "Green":
            return same
        if A == "Green" and B == "Ivory":
            return a - 1 == b
        if recurse == 0:
            return zebra_constraint(B, b, A, a, 1)
        if (
            (A in Colors and B in Colors)
            or (A in Pets and B in Pets)
            or (A in Drinks and B in Drinks)
            or (A in Countries and B in Countries)
            or (A in Smokes and B in Smokes)
        ):
            return not same
        raise Exception("error")

    return CSP(variables, domains, neighbors, zebra_constraint)


def solve_zebra(algorithm=min_conflicts, **args):
    """Solve the Zebra Puzzle and display the solution.

    Creates a Zebra Puzzle instance, solves it with the specified algorithm,
    and prints a human-readable solution showing which attributes belong
    to each house.

    Args:
        algorithm: CSP solving function to use (default: min_conflicts)
        **args: Additional arguments to pass to the solving algorithm

    Returns:
        tuple: (zebra_house, water_house, assignments_made, full_solution)
            - zebra_house: House number (1-5) where zebra is kept
            - water_house: House number (1-5) where water is drunk
            - assignments_made: Number of variable assignments during search
            - full_solution: Complete assignment dictionary
    """
    z = Zebra()
    ans = algorithm(z, **args)
    for h in range(1, 6):
        print("House", h, end=" ")
        for var, val in ans.items():
            if val == h:
                print(var, end=" ")
        print()
    return ans["Zebra"], ans["Water"], z.nassigns, ans


# ______________________________________________________________________________
# n-ary Constraint Satisfaction Problem


class NaryCSP:
    """
    A nary-CSP consists of:
    domains     : a dictionary that maps each variable to its domain
    constraints : a list of constraints
    variables   : a set of variables
    var_to_const: a variable to set of constraints dictionary
    """

    def __init__(self, domains, constraints):
        """Domains is a variable:domain dictionary
        constraints is a list of constraints
        """
        self.variables = set(domains)
        self.domains = domains
        self.constraints = constraints
        self.var_to_const = {var: set() for var in self.variables}
        for con in constraints:
            for var in con.scope:
                self.var_to_const[var].add(con)

    def __str__(self):
        """String representation of CSP"""
        return str(self.domains)

    def display(self, assignment=None):
        """More detailed string representation of CSP"""
        if assignment is None:
            assignment = {}
        print(assignment)

    def consistent(self, assignment):
        """assignment is a variable:value dictionary
        returns True if all of the constraints that can be evaluated
                        evaluate to True given assignment.
        """
        return all(
            con.holds(assignment)
            for con in self.constraints
            if all(v in assignment for v in con.scope)
        )


class Constraint:
    """
    A Constraint consists of:
    scope    : a tuple of variables
    condition: a function that can applied to a tuple of values
    for the variables.
    """

    def __init__(self, scope, condition):
        self.scope = scope
        self.condition = condition

    def __repr__(self):
        return self.condition.__name__ + str(self.scope)

    def holds(self, assignment):
        """Returns the value of Constraint con evaluated in assignment.

        precondition: all variables are assigned in assignment
        """
        return self.condition(*tuple(assignment[v] for v in self.scope))


def all_diff_constraint(*values):
    """Returns True if all values are different, False otherwise.

    A constraint ensuring that all variables in a group have different values.
    Commonly used in puzzles like Sudoku or N-Queens.

    Args:
        *values: Variable number of values to check for uniqueness

    Returns:
        bool: True if all values are unique, False if any duplicates exist

    Example:
        >>> all_diff_constraint(1, 2, 3)
        True
        >>> all_diff_constraint(1, 2, 1)
        False
    """
    return len(values) is len(set(values))


def is_word_constraint(words):
    """Returns True if the letters concatenated form a word in words, False otherwise.

    Creates a constraint function that checks if a sequence of letters
    forms a valid word from the given word set. Used in crossword puzzles.

    Args:
        words: Set or collection of valid words

    Returns:
        function: Constraint function that takes letter sequence and returns bool

    Example:
        >>> word_check = is_word_constraint({'cat', 'dog'})
        >>> word_check('c', 'a', 't')
        True
        >>> word_check('x', 'y', 'z')
        False
    """

    def isw(*letters):
        return "".join(letters) in words

    return isw


def meet_at_constraint(p1, p2):
    """Returns a function that is True when the words meet at the positions (p1, p2), False otherwise.

    Creates a constraint for crossword puzzles where two words intersect.
    The constraint ensures that the letter at position p1 in the first word
    matches the letter at position p2 in the second word.

    Args:
        p1: Position in first word (0-indexed)
        p2: Position in second word (0-indexed)

    Returns:
        function: Constraint function that checks if words intersect correctly

    Example:
        >>> intersect = meet_at_constraint(2, 0)  # 3rd letter of w1 = 1st letter of w2
        >>> intersect('cat', 'top')  # 't' == 't'
        True
        >>> intersect('cat', 'dog')  # 't' != 'd'
        False
    """

    def meets(w1, w2):
        return w1[p1] == w2[p2]

    meets.__name__ = "meet_at(" + str(p1) + "," + str(p2) + ")"
    return meets


def adjacent_constraint(x, y):
    """Returns True if x and y are adjacent numbers, False otherwise.

    Constraint that ensures two values differ by exactly 1. Useful for
    problems where variables must have neighboring values.

    Args:
        x: First numeric value
        y: Second numeric value

    Returns:
        bool: True if |x - y| == 1, False otherwise

    Example:
        >>> adjacent_constraint(3, 4)
        True
        >>> adjacent_constraint(3, 5)
        False
    """
    return abs(x - y) == 1


def sum_constraint(n):
    """Create a constraint that requires values to sum to exactly n.

    Returns a function that checks if a group of values sums to the target.
    Commonly used in arithmetic puzzles like Kakuro or magic squares.

    Args:
        n: Target sum value

    Returns:
        function: Constraint function that takes multiple values and returns bool

    Example:
        >>> sum_to_10 = sum_constraint(10)
        >>> sum_to_10(3, 4, 3)
        True   # 3 + 4 + 3 = 10
        >>> sum_to_10(1, 2, 3)
        False  # 1 + 2 + 3 = 6 ≠ 10

    Note: This function uses 'is' instead of '==' which is unusual but intentional
          for this specific implementation style.
    """

    def sumv(*values):
        return sum(values) is n  # Note: 'is' used instead of '==' here

    sumv.__name__ = str(n) + "==sum"  # Give the function a descriptive name
    return sumv


def is_constraint(val):
    """Returns a function that is True when x is equal to val, False otherwise"""

    def isv(x):
        return val == x

    isv.__name__ = str(val) + "=="
    return isv


def ne_constraint(val):
    """Returns a function that is True when x is not equal to val, False otherwise"""

    def nev(x):
        return val != x

    nev.__name__ = str(val) + "!="
    return nev


def no_heuristic(to_do):
    """Default heuristic that returns the to_do list unchanged.

    Args:
        to_do: Collection of items to be processed

    Returns:
        The unmodified to_do collection
    """
    return to_do


def sat_up(to_do):
    """Heuristic for ordering constraint-variable pairs by constraint scope size.

    Orders constraints by the inverse of their scope size, prioritizing
    constraints that involve fewer variables (tighter constraints first).

    Args:
        to_do: Collection of (variable, constraint) pairs

    Returns:
        SortedSet ordered by constraint scope size (smallest first)
    """
    return SortedSet(to_do, key=lambda t: 1 / len([var for var in t[1].scope]))


class ACSolver:
    """Solves a CSP with arc consistency and domain splitting"""

    def __init__(self, csp):
        """a CSP solver that uses arc consistency
        * csp is the CSP to be solved
        """
        self.csp = csp

    def GAC(self, orig_domains=None, to_do=None, arc_heuristic=sat_up):
        """
        Makes this CSP arc-consistent using Generalized Arc Consistency
        orig_domains: is the original domains
        to_do       : is a set of (variable,constraint) pairs
        returns the reduced domains (an arc-consistent variable:domain dictionary)
        """
        if orig_domains is None:
            orig_domains = self.csp.domains
        if to_do is None:
            to_do = {
                (var, const) for const in self.csp.constraints for var in const.scope
            }
        else:
            to_do = to_do.copy()
        domains = orig_domains.copy()
        to_do = arc_heuristic(to_do)
        checks = 0
        while to_do:
            var, const = to_do.pop()
            other_vars = [ov for ov in const.scope if ov != var]
            new_domain = set()
            if len(other_vars) == 0:
                for val in domains[var]:
                    if const.holds({var: val}):
                        new_domain.add(val)
                    checks += 1
                # Alternative implementation using set comprehension:
                # new_domain = {val for val in domains[var]
                #               if const.holds({var: val})}
            elif len(other_vars) == 1:
                other = other_vars[0]
                for val in domains[var]:
                    for other_val in domains[other]:
                        checks += 1
                        if const.holds({var: val, other: other_val}):
                            new_domain.add(val)
                            break
                # Alternative implementation using set comprehension:
                # new_domain = {val for val in domains[var]
                #               if any(const.holds({var: val, other: other_val})
                #                      for other_val in domains[other])}
            else:  # general case
                for val in domains[var]:
                    holds, checks = self.any_holds(
                        domains, const, {var: val}, other_vars, checks=checks
                    )
                    if holds:
                        new_domain.add(val)
                # Alternative implementation using set comprehension:
                # new_domain = {val for val in domains[var]
                #               if self.any_holds(domains, const, {var: val}, other_vars)}
            if new_domain != domains[var]:
                domains[var] = new_domain
                if not new_domain:
                    return False, domains, checks
                add_to_do = self.new_to_do(var, const).difference(to_do)
                to_do |= add_to_do
        return True, domains, checks

    def new_to_do(self, var, const):
        """
        Returns new elements to be added to to_do after assigning
        variable var in constraint const.
        """
        return {
            (nvar, nconst)
            for nconst in self.csp.var_to_const[var]
            if nconst != const
            for nvar in nconst.scope
            if nvar != var
        }

    def any_holds(self, domains, const, env, other_vars, ind=0, checks=0):
        """
        Returns True if Constraint const holds for an assignment
        that extends env with the variables in other_vars[ind:]
        env is a dictionary
        Warning: this has side effects and changes the elements of env
        """
        if ind == len(other_vars):
            return const.holds(env), checks + 1
        else:
            var = other_vars[ind]
            for val in domains[var]:
                # Alternative approach with no side effects:
                # env = dict_union(env, {var:val})  # no side effects
                env[var] = val
                holds, checks = self.any_holds(
                    domains, const, env, other_vars, ind + 1, checks
                )
                if holds:
                    return True, checks
            return False, checks

    def domain_splitting(self, domains=None, to_do=None, arc_heuristic=sat_up):
        """
        Return a solution to the current CSP or False if there are no solutions
        to_do is the list of arcs to check
        """
        if domains is None:
            domains = self.csp.domains
        consistency, new_domains, _ = self.GAC(domains, to_do, arc_heuristic)
        if not consistency:
            return False
        elif all(len(new_domains[var]) == 1 for var in domains):
            return {var: first(new_domains[var]) for var in domains}
        else:
            var = first(x for x in self.csp.variables if len(new_domains[x]) > 1)
            if var:
                dom1, dom2 = partition_domain(new_domains[var])
                new_doms1 = extend(new_domains, var, dom1)
                new_doms2 = extend(new_domains, var, dom2)
                to_do = self.new_to_do(var, None)
                return self.domain_splitting(
                    new_doms1, to_do, arc_heuristic
                ) or self.domain_splitting(new_doms2, to_do, arc_heuristic)


def partition_domain(dom):
    """Partitions domain dom into two roughly equal subsets.

    Splits a domain into two parts for domain splitting search strategies.
    Used by algorithms that recursively split domains to solve CSPs.

    Args:
        dom: Set or sequence representing a variable's domain

    Returns:
        tuple: (dom1, dom2) where dom1 and dom2 partition the original domain
               dom1 contains first half, dom2 contains second half
    """
    split = len(dom) // 2
    dom1 = set(list(dom)[:split])
    dom2 = dom - dom1
    return dom1, dom2


class ACSearchSolver(search.Problem):
    """A search problem with arc consistency and domain splitting
    A node is a CSP"""

    def __init__(self, csp, arc_heuristic=sat_up):
        self.cons = ACSolver(csp)
        consistency, self.domains, _ = self.cons.GAC(arc_heuristic=arc_heuristic)
        if not consistency:
            raise Exception("CSP is inconsistent")
        self.heuristic = arc_heuristic
        super().__init__(self.domains)

    def goal_test(self, node):
        """Node is a goal if all domains have 1 element"""
        return all(len(node[var]) == 1 for var in node)

    def actions(self, state):
        var = first(x for x in state if len(state[x]) > 1)
        neighs = []
        if var:
            dom1, dom2 = partition_domain(state[var])
            to_do = self.cons.new_to_do(var, None)
            for dom in [dom1, dom2]:
                new_domains = extend(state, var, dom)
                consistency, cons_doms, _ = self.cons.GAC(
                    new_domains, to_do, self.heuristic
                )
                if consistency:
                    neighs.append(cons_doms)
        return neighs

    def result(self, state, action):
        return action


def ac_solver(csp, arc_heuristic=sat_up):
    """Arc consistency solver using domain splitting approach.

    Solves CSPs by maintaining arc consistency and recursively splitting
    domains when multiple values remain. More efficient than backtracking
    for some problem types.

    Args:
        csp: The constraint satisfaction problem to solve
        arc_heuristic: Heuristic for ordering constraint processing

    Returns:
        dict or False: Solution assignment if found, False if unsolvable
    """
    return ACSolver(csp).domain_splitting(arc_heuristic=arc_heuristic)


def ac_search_solver(csp, arc_heuristic=sat_up):
    """Arc consistency solver using search-based approach.

    Alternative implementation that uses depth-first search combined
    with arc consistency and domain splitting. Provides a search-based
    interface to the arc consistency solving approach.

    Args:
        csp: The constraint satisfaction problem to solve
        arc_heuristic: Heuristic for ordering constraint processing

    Returns:
        dict or None: Solution assignment if found, None if unsolvable

    Note: Uses ACSearchSolver internally with depth_first_tree_search
    """
    from search import depth_first_tree_search

    solution = None
    try:
        solution = depth_first_tree_search(
            ACSearchSolver(csp, arc_heuristic=arc_heuristic)
        ).state
    except:
        return solution
    if solution:
        return {var: first(solution[var]) for var in solution}


# ______________________________________________________________________________
# Crossword Problem


csp_crossword = NaryCSP(
    {
        "one_across": {"ant", "big", "bus", "car", "has"},
        "one_down": {"book", "buys", "hold", "lane", "year"},
        "two_down": {"ginger", "search", "symbol", "syntax"},
        "three_across": {"book", "buys", "hold", "land", "year"},
        "four_across": {"ant", "big", "bus", "car", "has"},
    },
    [
        Constraint(("one_across", "one_down"), meet_at_constraint(0, 0)),
        Constraint(("one_across", "two_down"), meet_at_constraint(2, 0)),
        Constraint(("three_across", "two_down"), meet_at_constraint(2, 2)),
        Constraint(("three_across", "one_down"), meet_at_constraint(0, 2)),
        Constraint(("four_across", "two_down"), meet_at_constraint(0, 4)),
    ],
)

crossword1 = [
    ["_", "_", "_", "*", "*"],
    ["_", "*", "_", "*", "*"],
    ["_", "_", "_", "_", "*"],
    ["_", "*", "_", "*", "*"],
    ["*", "*", "_", "_", "_"],
    ["*", "*", "_", "*", "*"],
]

words1 = {
    "ant",
    "big",
    "bus",
    "car",
    "has",
    "book",
    "buys",
    "hold",
    "lane",
    "year",
    "ginger",
    "search",
    "symbol",
    "syntax",
}


class Crossword(NaryCSP):
    """CSP for crossword puzzle construction.

    Models crossword puzzles as CSPs where variables represent letter positions
    and constraints ensure that sequences of letters form valid words from
    a given word list.

    Args:
        puzzle: 2D grid with '_' for letters, '*' for black squares
        words: Set of valid words for the crossword
    """

    def __init__(self, puzzle, words):
        domains = {}
        constraints = []
        for i, line in enumerate(puzzle):
            scope = []
            for j, element in enumerate(line):
                if element == "_":
                    var = "p" + str(j) + str(i)
                    domains[var] = list(string.ascii_lowercase)
                    scope.append(var)
                else:
                    if len(scope) > 1:
                        constraints.append(
                            Constraint(tuple(scope), is_word_constraint(words))
                        )
                    scope.clear()
            if len(scope) > 1:
                constraints.append(Constraint(tuple(scope), is_word_constraint(words)))
        puzzle_t = list(map(list, zip(*puzzle)))
        for i, line in enumerate(puzzle_t):
            scope = []
            for j, element in enumerate(line):
                if element == "_":
                    scope.append("p" + str(i) + str(j))
                else:
                    if len(scope) > 1:
                        constraints.append(
                            Constraint(tuple(scope), is_word_constraint(words))
                        )
                    scope.clear()
            if len(scope) > 1:
                constraints.append(Constraint(tuple(scope), is_word_constraint(words)))
        super().__init__(domains, constraints)
        self.puzzle = puzzle

    def display(self, assignment=None):
        for i, line in enumerate(self.puzzle):
            puzzle = ""
            for j, element in enumerate(line):
                if element == "*":
                    puzzle += "[*] "
                else:
                    var = "p" + str(j) + str(i)
                    if assignment is not None:
                        if (
                            isinstance(assignment[var], set)
                            and len(assignment[var]) == 1
                        ):
                            puzzle += "[" + str(first(assignment[var])).upper() + "] "
                        elif isinstance(assignment[var], str):
                            puzzle += "[" + str(assignment[var]).upper() + "] "
                        else:
                            puzzle += "[_] "
                    else:
                        puzzle += "[_] "
            print(puzzle)


# ______________________________________________________________________________
# Kakuro Problem


# difficulty 0
kakuro1 = [
    ["*", "*", "*", [6, ""], [3, ""]],
    ["*", [4, ""], [3, 3], "_", "_"],
    [["", 10], "_", "_", "_", "_"],
    [["", 3], "_", "_", "*", "*"],
]

# difficulty 0
kakuro2 = [
    ["*", [10, ""], [13, ""], "*"],
    [["", 3], "_", "_", [13, ""]],
    [["", 12], "_", "_", "_"],
    [["", 21], "_", "_", "_"],
]

# difficulty 1
kakuro3 = [
    ["*", [17, ""], [28, ""], "*", [42, ""], [22, ""]],
    [["", 9], "_", "_", [31, 14], "_", "_"],
    [["", 20], "_", "_", "_", "_", "_"],
    ["*", ["", 30], "_", "_", "_", "_"],
    ["*", [22, 24], "_", "_", "_", "*"],
    [["", 25], "_", "_", "_", "_", [11, ""]],
    [["", 20], "_", "_", "_", "_", "_"],
    [["", 14], "_", "_", ["", 17], "_", "_"],
]

# difficulty 2
kakuro4 = [
    [
        "*",
        "*",
        "*",
        "*",
        "*",
        [4, ""],
        [24, ""],
        [11, ""],
        "*",
        "*",
        "*",
        [11, ""],
        [17, ""],
        "*",
        "*",
    ],
    [
        "*",
        "*",
        "*",
        [17, ""],
        [11, 12],
        "_",
        "_",
        "_",
        "*",
        "*",
        [24, 10],
        "_",
        "_",
        [11, ""],
        "*",
    ],
    [
        "*",
        [4, ""],
        [16, 26],
        "_",
        "_",
        "_",
        "_",
        "_",
        "*",
        ["", 20],
        "_",
        "_",
        "_",
        "_",
        [16, ""],
    ],
    [
        ["", 20],
        "_",
        "_",
        "_",
        "_",
        [24, 13],
        "_",
        "_",
        [16, ""],
        ["", 12],
        "_",
        "_",
        [23, 10],
        "_",
        "_",
    ],
    [
        ["", 10],
        "_",
        "_",
        [24, 12],
        "_",
        "_",
        [16, 5],
        "_",
        "_",
        [16, 30],
        "_",
        "_",
        "_",
        "_",
        "_",
    ],
    [
        "*",
        "*",
        [3, 26],
        "_",
        "_",
        "_",
        "_",
        ["", 12],
        "_",
        "_",
        [4, ""],
        [16, 14],
        "_",
        "_",
        "*",
    ],
    [
        "*",
        ["", 8],
        "_",
        "_",
        ["", 15],
        "_",
        "_",
        [34, 26],
        "_",
        "_",
        "_",
        "_",
        "_",
        "*",
        "*",
    ],
    [
        "*",
        ["", 11],
        "_",
        "_",
        [3, ""],
        [17, ""],
        ["", 14],
        "_",
        "_",
        ["", 8],
        "_",
        "_",
        [7, ""],
        [17, ""],
        "*",
    ],
    [
        "*",
        "*",
        "*",
        [23, 10],
        "_",
        "_",
        [3, 9],
        "_",
        "_",
        [4, ""],
        [23, ""],
        ["", 13],
        "_",
        "_",
        "*",
    ],
    [
        "*",
        "*",
        [10, 26],
        "_",
        "_",
        "_",
        "_",
        "_",
        ["", 7],
        "_",
        "_",
        [30, 9],
        "_",
        "_",
        "*",
    ],
    [
        "*",
        [17, 11],
        "_",
        "_",
        [11, ""],
        [24, 8],
        "_",
        "_",
        [11, 21],
        "_",
        "_",
        "_",
        "_",
        [16, ""],
        [17, ""],
    ],
    [
        ["", 29],
        "_",
        "_",
        "_",
        "_",
        "_",
        ["", 7],
        "_",
        "_",
        [23, 14],
        "_",
        "_",
        [3, 17],
        "_",
        "_",
    ],
    [
        ["", 10],
        "_",
        "_",
        [3, 10],
        "_",
        "_",
        "*",
        ["", 8],
        "_",
        "_",
        [4, 25],
        "_",
        "_",
        "_",
        "_",
    ],
    [
        "*",
        ["", 16],
        "_",
        "_",
        "_",
        "_",
        "*",
        ["", 23],
        "_",
        "_",
        "_",
        "_",
        "_",
        "*",
        "*",
    ],
    [
        "*",
        "*",
        ["", 6],
        "_",
        "_",
        "*",
        "*",
        ["", 15],
        "_",
        "_",
        "_",
        "*",
        "*",
        "*",
        "*",
    ],
]


class Kakuro(NaryCSP):
    """CSP for Kakuro number puzzles.

    Models Kakuro puzzles as CSPs where variables represent cell values (1-9)
    and constraints ensure that groups of cells sum to specified values
    while maintaining all-different constraints within each group.

    Args:
        puzzle: 2D grid with '_' for cells, '*' for black squares,
               and [down_sum, right_sum] for clue cells
    """

    def __init__(self, puzzle):
        variables = []
        for i, line in enumerate(puzzle):
            # Create variables for each empty cell position
            for j, element in enumerate(line):
                if element == "_":
                    # Variable naming: "X" + zero-padded row + zero-padded column (e.g., "X0203")
                    var1 = str(i)
                    if len(var1) == 1:
                        var1 = "0" + var1
                    var2 = str(j)
                    if len(var2) == 1:
                        var2 = "0" + var2
                    variables.append("X" + var1 + var2)
        domains = {}
        for var in variables:
            domains[var] = set(range(1, 10))
        constraints = []
        for i, line in enumerate(puzzle):
            for j, element in enumerate(line):
                if element != "_" and element != "*":
                    # down - column
                    if element[0] != "":
                        x = []
                        for k in range(i + 1, len(puzzle)):
                            if puzzle[k][j] != "_":
                                break
                            var1 = str(k)
                            if len(var1) == 1:
                                var1 = "0" + var1
                            var2 = str(j)
                            if len(var2) == 1:
                                var2 = "0" + var2
                            x.append("X" + var1 + var2)
                        constraints.append(Constraint(x, sum_constraint(element[0])))
                        constraints.append(Constraint(x, all_diff_constraint))
                    # right - line
                    if element[1] != "":
                        x = []
                        for k in range(j + 1, len(puzzle[i])):
                            if puzzle[i][k] != "_":
                                break
                            var1 = str(i)
                            if len(var1) == 1:
                                var1 = "0" + var1
                            var2 = str(k)
                            if len(var2) == 1:
                                var2 = "0" + var2
                            x.append("X" + var1 + var2)
                        constraints.append(Constraint(x, sum_constraint(element[1])))
                        constraints.append(Constraint(x, all_diff_constraint))
        super().__init__(domains, constraints)
        self.puzzle = puzzle

    def display(self, assignment=None):
        for i, line in enumerate(self.puzzle):
            puzzle = ""
            for j, element in enumerate(line):
                if element == "*":
                    puzzle += "[*]\t"
                elif element == "_":
                    var1 = str(i)
                    if len(var1) == 1:
                        var1 = "0" + var1
                    var2 = str(j)
                    if len(var2) == 1:
                        var2 = "0" + var2
                    var = "X" + var1 + var2
                    if assignment is not None:
                        if (
                            isinstance(assignment[var], set)
                            and len(assignment[var]) == 1
                        ):
                            puzzle += "[" + str(first(assignment[var])) + "]\t"
                        elif isinstance(assignment[var], int):
                            puzzle += "[" + str(assignment[var]) + "]\t"
                        else:
                            puzzle += "[_]\t"
                    else:
                        puzzle += "[_]\t"
                else:
                    puzzle += str(element[0]) + "\\" + str(element[1]) + "\t"
            print(puzzle)


# ______________________________________________________________________________
# Cryptarithmetic Problem

# [Figure 6.2]
# T W O + T W O = F O U R
two_two_four = NaryCSP(
    {
        "T": set(range(1, 10)),
        "F": set(range(1, 10)),
        "W": set(range(0, 10)),
        "O": set(range(0, 10)),
        "U": set(range(0, 10)),
        "R": set(range(0, 10)),
        "C1": set(range(0, 2)),
        "C2": set(range(0, 2)),
        "C3": set(range(0, 2)),
    },
    [
        Constraint(("T", "F", "W", "O", "U", "R"), all_diff_constraint),
        Constraint(("O", "R", "C1"), lambda o, r, c1: o + o == r + 10 * c1),
        Constraint(
            ("W", "U", "C1", "C2"), lambda w, u, c1, c2: c1 + w + w == u + 10 * c2
        ),
        Constraint(
            ("T", "O", "C2", "C3"), lambda t, o, c2, c3: c2 + t + t == o + 10 * c3
        ),
        Constraint(("F", "C3"), eq),
    ],
)

# S E N D + M O R E = M O N E Y
send_more_money = NaryCSP(
    {
        "S": set(range(1, 10)),
        "M": set(range(1, 10)),
        "E": set(range(0, 10)),
        "N": set(range(0, 10)),
        "D": set(range(0, 10)),
        "O": set(range(0, 10)),
        "R": set(range(0, 10)),
        "Y": set(range(0, 10)),
        "C1": set(range(0, 2)),
        "C2": set(range(0, 2)),
        "C3": set(range(0, 2)),
        "C4": set(range(0, 2)),
    },
    [
        Constraint(("S", "E", "N", "D", "M", "O", "R", "Y"), all_diff_constraint),
        Constraint(("D", "E", "Y", "C1"), lambda d, e, y, c1: d + e == y + 10 * c1),
        Constraint(
            ("N", "R", "E", "C1", "C2"),
            lambda n, r, e, c1, c2: c1 + n + r == e + 10 * c2,
        ),
        Constraint(
            ("E", "O", "N", "C2", "C3"),
            lambda e, o, n, c2, c3: c2 + e + o == n + 10 * c3,
        ),
        Constraint(
            ("S", "M", "O", "C3", "C4"),
            lambda s, m, o, c3, c4: c3 + s + m == o + 10 * c4,
        ),
        Constraint(("M", "C4"), eq),
    ],
)
