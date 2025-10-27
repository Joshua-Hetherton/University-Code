"""
Intelligent Agents and Environments Framework (AIMA Chapters 1-2)

This module implements the foundational concepts of Artificial Intelligence:
- Agents: Entities that perceive their environment and take actions
- Environments: The world in which agents operate and interact

=== CORE AI CONCEPTS IMPLEMENTED ===

PEAS Framework (Performance, Environment, Actuators, Sensors):
- Performance: Measured through agent.performance attribute
- Environment: Simulated worlds where agents operate
- Actuators: Actions agents can perform (movement, grabbing objects)
- Sensors: Percepts agents receive from their environment

AGENT TYPES (from simple to complex):
1. Simple Reflex Agents: React based on current percept only
2. Model-Based Agents: Maintain internal state/memory
3. Goal-Based Agents: Work toward specific objectives
4. Utility-Based Agents: Optimize performance measures

=== CLASS HIERARCHIES ===

Thing (Base class for all physical objects)
├── Agent (Intelligent entities that can act)
│   ├── Wumpus (Monster in Wumpus World)
│   └── Explorer (Player agent in Wumpus World)
├── Dirt (Object to be cleaned by vacuum agents)
├── Wall (Obstacle that blocks movement)
├── Gold (Treasure in Wumpus World)
└── ... (other environment objects)

Environment (Abstract base for all environments)
├── XYEnvironment (2D grid-based worlds)
│   ├── VacuumEnvironment (Cleaning simulation)
│   ├── WumpusEnvironment (Classic AI problem)
│   └── GraphicEnvironment (Visual representation)
└── TrivialVacuumEnvironment (Simple 2-location world)

Agent Programs (The "brains" of agents):
- SimpleReflexAgentProgram: if-then rules
- ModelBasedReflexAgentProgram: maintains world model
- TableDrivenAgentProgram: lookup table approach
- RandomAgentProgram: random action selection

=== USAGE FOR STUDENTS ===
This framework lets you:
1. Create custom agents with different decision-making strategies
2. Design environments with varying complexity
3. Compare agent performance across different scenarios
4. Visualize agent behavior in real-time

Example: Creating a simple cleaning robot
```python
agent = ReflexVacuumAgent()           # Create agent
env = TrivialVacuumEnvironment()      # Create world
env.add_thing(agent)                  # Place agent in world
env.run(100)                          # Run simulation
print(f"Score: {agent.performance}") # Check results
```
"""

# TODO: GUI speed control needs fixing - currently has no effect on simulation speed

from utils import distance_squared, turn_heading
from statistics import mean
from ipythonblocks import BlockGrid
from IPython.display import HTML, display, clear_output
from time import sleep

import random
import copy
import collections
import numbers


# ================================================================================
# CORE CLASSES: THINGS AND AGENTS
# ================================================================================


class Thing:
    """
    Base class for all physical objects that can exist in an environment.

    In AI, we model worlds as containing "things" - objects that can be perceived,
    manipulated, or interacted with by agents. This includes both passive objects
    (like dirt or walls) and active agents (like robots or characters).

    Attributes:
        __name__ (optional): Human-readable name for display purposes
        location: Position in the environment (set by Environment.add_thing())

    Note for students: Think of Thing as the "atom" of our simulated world.
    Everything that exists in an AI environment inherits from Thing.
    """

    def __repr__(self):
        """Return string representation of this Thing for debugging/display."""
        return "<{}>".format(getattr(self, "__name__", self.__class__.__name__))

    def is_alive(self):
        """
        Check if this Thing is considered 'alive' (for agents).

        Returns:
            bool: True if thing has 'alive' attribute set to True

        Note: Only meaningful for Agent subclasses. Used to determine
        if an agent can still act or if simulation should continue.
        """
        return hasattr(self, "alive") and self.alive

    def show_state(self):
        """
        Display the internal state of this Thing.

        Default implementation provides placeholder message.
        Subclasses should override to show meaningful state information
        (e.g., agent's knowledge, held objects, etc.).
        """
        print("I don't know how to show_state.")

    def display(self, canvas, x, y, width, height):
        """
        Render this Thing on a graphical canvas.

        Args:
            canvas: Graphics canvas object
            x, y: Position coordinates
            width, height: Rendering dimensions

        Note: Currently unimplemented. Override in subclasses that need
        custom graphical representation.
        """
        # Do we need this?
        pass


class Agent(Thing):
    """
    An intelligent agent that perceives its environment and takes actions.

    This is the core concept in AI! An agent is anything that:
    1. Perceives its environment through sensors (gets percepts)
    2. Processes information using its "program" (decision-making function)
    3. Acts on the environment through actuators (performs actions)

    Mathematical Model: Agent = f(percept_sequence) → action

    Key Attributes:
        program (callable): The agent's "brain" - maps percepts to actions
            Signature: program(percept) → action
        alive (bool): Whether agent is still active/functional
        bump (bool): Collision detection flag (set by environment)
        holding (list): Objects currently carried by the agent
        performance (float): Cumulative performance score
        location: Current position in the environment

    Design Principle: The program function can ONLY use percept information
    (no cheating by accessing internal environment state). This forces
    agents to build their own world models from observations.

    For students: This class implements the fundamental agent-environment
    interaction loop that drives all intelligent behavior in AI systems.
    """

    def __init__(self, program=None):
        # Initialize agent's state variables
        self.alive = True  # Agent starts active and functional
        self.bump = False  # No collision detected initially
        self.holding = []  # Empty hands/inventory
        self.performance = 0  # Zero starting score

        # Set up the agent's decision-making program
        if program is None or not isinstance(program, collections.abc.Callable):
            print(
                "Can't find a valid program for {}, falling back to default.".format(
                    self.__class__.__name__
                )
            )

            def program(percept):
                # Default: Ask human for input (manual control mode)
                # This is useful for testing environments or educational demos
                return eval(input("Percept={}; action? ".format(percept)))

        self.program = program

    def can_grab(self, thing):
        """
        Determine if this agent can pick up/grab the specified object.

        Args:
            thing (Thing): Object the agent wants to grab

        Returns:
            bool: True if agent can grab this thing, False otherwise

        Default implementation: agents can't grab anything (like a basic robot
        without manipulator arms). Override in subclasses to define what
        each agent type can interact with.

        Example: VacuumAgent might grab Dirt, Explorer might grab Gold
        """
        return False


def TraceAgent(agent):
    """
    Debugging wrapper that logs agent's perception-action cycle.

    This is extremely useful for understanding how agents behave!
    It wraps an existing agent to print what it perceives and what
    action it chooses, giving you insight into the agent's decision process.

    Args:
        agent (Agent): The agent to wrap with logging

    Returns:
        Agent: Same agent but with logging enabled

    Example output:
        "VacuumAgent perceives [Dirt] and does Suck"
        "VacuumAgent perceives [] and does Forward"

    For students: Use this whenever you want to see what your agent is
    thinking! Essential for debugging agent behavior.
    """
    old_program = agent.program

    def new_program(percept):
        action = old_program(percept)
        print("{} perceives {} and does {}".format(agent, percept, action))
        return action

    agent.program = new_program
    return agent


# ================================================================================
# AGENT PROGRAM TYPES (The "Brains" of Different Agent Architectures)
# ================================================================================


def TableDrivenAgentProgram(table):
    """
    [Figure 2.7] Table-Driven Agent: Lookup table for percept sequences → actions

    This is the most basic agent architecture! It uses a precomputed table
    that maps every possible sequence of percepts to an action. Simple but
    has major limitations.

    Args:
        table (dict): Maps percept sequences to actions
            Format: {(percept1, percept2, ...): action}

    Returns:
        function: Agent program that can be used with Agent()

    Advantages:
        - Theoretically can implement any agent behavior
        - Easy to understand and verify

    Disadvantages:
        - Exponential memory requirements O(|Percepts|^t)
        - Impractical for any realistic environment
        - No learning or adaptation

    Time Complexity: O(1) per action (after table lookup)
    Space Complexity: O(|Percepts|^sequence_length) - EXPONENTIAL!

    Educational note: This shows why we need smarter agent architectures!
    """
    percepts = []  # Remember all percepts seen so far

    def program(percept):
        percepts.append(percept)
        action = table.get(tuple(percepts))
        return action

    return program


def RandomAgentProgram(actions):
    """
    Random Agent: Chooses actions randomly, ignoring all percepts.

    This might seem silly, but random agents are actually useful for:
    1. Baseline comparison (how well should a "smart" agent do vs random?)
    2. Exploration in unknown environments
    3. Testing environment robustness
    4. Simple behavior when sophisticated reasoning isn't needed

    Args:
        actions (list): Available actions the agent can choose from

    Returns:
        function: Agent program that randomly selects actions

    Performance: Usually poor, but provides lower bound for comparison

    Example:
        >>> actions = ['Right', 'Left', 'Suck', 'NoOp']
        >>> program = RandomAgentProgram(actions)
        >>> agent = Agent(program)
        >>> # Agent will now act randomly in any environment

    Note: Even random agents can sometimes succeed due to luck!
    """
    return lambda percept: random.choice(actions)


# ================================================================================
# REFLEX AGENT ARCHITECTURES (Condition-Action Rules)
# ================================================================================


def SimpleReflexAgentProgram(rules, interpret_input):
    """
    [Figure 2.10] Simple Reflex Agent: IF-THEN rules based on current percept

    This agent architecture uses condition-action rules that fire based on
    the current percept only (no memory of past). Think of it like reflexes
    in biology - immediate responses to stimuli.

    Architecture:
        Percept → Interpret → Match Rule → Action

    Args:
        rules (list): Collection of IF-THEN rules
            Each rule should have: rule.matches(state) and rule.action
        interpret_input (function): Converts percept to internal state representation

    Returns:
        function: Agent program implementing reflex behavior

    Advantages:
        - Fast execution (just pattern matching)
        - Simple to understand and implement
        - Works well in fully observable, deterministic environments

    Limitations:
        - No memory (can get stuck in loops)
        - Can't handle partial observability
        - No learning or adaptation

    Example use case: Thermostat (if temp > 70, turn on AC)
    """

    def program(percept):
        state = interpret_input(percept)
        rule = rule_match(state, rules)
        action = rule.action
        return action

    return program


def ModelBasedReflexAgentProgram(rules, update_state, model):
    """
    [Figure 2.12] Model-Based Reflex Agent: Uses internal state + world model

    This is a major step up from simple reflex agents! It maintains an internal
    model of the world state and updates it based on percepts and actions.
    This allows handling partial observability and avoiding infinite loops.

    Architecture:
        Percept + Action → Update State → Match Rule → Action

    Args:
        rules (list): Condition-action rules (same as SimpleReflexAgent)
        update_state (function): Updates internal state based on:
            (old_state, action, percept, model) → new_state
        model (object): Knowledge about how the world works

    Returns:
        function: Agent program with memory and world modeling

    Key Innovation: Internal state lets agent remember:
        - Where it's been (avoid revisiting locations)
        - What it's accomplished (track progress)
        - What it knows about unseen parts of environment

    Example: Vacuum agent remembers which rooms are clean/dirty
    even when not currently observing them.
    """

    def program(percept):
        # Update our model of the world based on what we perceive and what we did
        program.state = update_state(program.state, program.action, percept, model)
        rule = rule_match(program.state, rules)
        action = rule.action
        program.action = action  # Remember our action for next update
        return action

    # Initialize state and action history
    program.state = program.action = None
    return program


def rule_match(state, rules):
    """
    Find the first rule whose condition matches the current state.

    Args:
        state: Current state representation
        rules (list): Collection of rules to check

    Returns:
        Rule object with matching condition, or None if no match

    Note: Uses first-match strategy. Rule order matters!
    More specific rules should come before general ones.
    """
    for rule in rules:
        if rule.matches(state):
            return rule


# ================================================================================
# VACUUM WORLD: Classic AI Environment for Learning Agent Concepts
# ================================================================================

# The two locations in the simple vacuum world
loc_A, loc_B = (0, 0), (1, 0)  # Location A at origin, Location B to the right


def RandomVacuumAgent():
    """
    Random Vacuum Agent: Baseline agent that acts randomly in vacuum world.

    Actions: 'Right', 'Left', 'Suck', 'NoOp'

    This provides a performance baseline - any "intelligent" agent should
    do better than random! Useful for comparing different agent strategies.

    Expected performance: Poor, since it may suck clean locations or
    move without purpose, but sometimes gets lucky.

    Educational value: Shows importance of using environmental information
    rather than ignoring it.
    """
    return Agent(RandomAgentProgram(["Right", "Left", "Suck", "NoOp"]))


def TableDrivenVacuumAgent():
    """
    [Figure 2.3] Table-Driven Vacuum Agent: Complete lookup table approach

    This agent has a precomputed table that maps every possible sequence
    of percepts to the optimal action. Demonstrates the table-driven approach
    from Figure 2.3 in the textbook.

    Percept format: (location, status) where:
        - location: loc_A or loc_B
        - status: 'Clean' or 'Dirty'

    The table handles sequences up to length 3, covering all possible
    scenarios in the two-location vacuum world.

    Performance: Optimal (since table was designed by hand for this environment)
    Problem: Table size grows exponentially with environment complexity!

    Educational note: Shows why table-driven agents don't scale to real problems.
    """
    # Hand-crafted optimal action table for 2-location vacuum world
    table = {
        ((loc_A, "Clean"),): "Right",
        ((loc_A, "Dirty"),): "Suck",
        ((loc_B, "Clean"),): "Left",
        ((loc_B, "Dirty"),): "Suck",
        ((loc_A, "Dirty"), (loc_A, "Clean")): "Right",
        ((loc_A, "Clean"), (loc_B, "Dirty")): "Suck",
        ((loc_B, "Clean"), (loc_A, "Dirty")): "Suck",
        ((loc_B, "Dirty"), (loc_B, "Clean")): "Left",
        ((loc_A, "Dirty"), (loc_A, "Clean"), (loc_B, "Dirty")): "Suck",
        ((loc_B, "Dirty"), (loc_B, "Clean"), (loc_A, "Dirty")): "Suck",
    }
    return Agent(TableDrivenAgentProgram(table))


def ReflexVacuumAgent():
    """
    [Figure 2.8] Simple Reflex Vacuum Agent: Rule-based cleaning robot

    This agent uses simple condition-action rules based only on current percept:
    1. If current location is dirty → Suck
    2. If at location A and clean → Move Right
    3. If at location B and clean → Move Left

    This demonstrates a Simple Reflex Agent architecture - no memory,
    just immediate responses to current situation.

    Advantages:
        - Fast and simple
        - Works well in this environment
        - Easy to understand and verify

    Limitations:
        - Could get stuck in infinite loops in other environments
        - No memory of what it has cleaned
        - Limited to fully observable environments

    Performance: Good in 2-location world, but would struggle in
    more complex environments without memory.
    """

    def program(percept):
        location, status = percept
        if status == "Dirty":
            return "Suck"
        elif location == loc_A:
            return "Right"
        elif location == loc_B:
            return "Left"

    return Agent(program)


def ModelBasedVacuumAgent():
    """
    Model-Based Vacuum Agent: Intelligent cleaning robot with memory

    This agent maintains an internal model of which locations are clean/dirty,
    allowing it to avoid unnecessary work and stop when the job is complete.

    Key improvements over ReflexVacuumAgent:
    1. Remembers the state of both locations
    2. Stops working when everything is clean (NoOp)
    3. Won't waste energy on already-clean locations

    Internal Model: {loc_A: status, loc_B: status}
        - status can be 'Clean', 'Dirty', or None (unknown)

    Decision Logic:
    1. If both locations known to be clean → NoOp (optimal stopping)
    2. If current location dirty → Suck
    3. Otherwise → Move to other location

    This demonstrates how memory/state can dramatically improve agent
    performance and efficiency!

    Performance: Optimal in 2-location world (cleans everything exactly once)
    """
    model = {loc_A: None, loc_B: None}  # Initially don't know status of either location

    def program(percept):
        """Enhanced vacuum logic with memory and optimal stopping."""
        location, status = percept
        model[location] = status  # Update our knowledge of current location

        # Optimal stopping: if everything is clean, we're done!
        if model[loc_A] == model[loc_B] == "Clean":
            return "NoOp"
        # Always clean dirty locations
        elif status == "Dirty":
            return "Suck"
        # Move to explore/clean other location
        elif location == loc_A:
            return "Right"
        elif location == loc_B:
            return "Left"

    return Agent(program)


# ================================================================================
# ENVIRONMENT FRAMEWORK: The World Where Agents Live and Act
# ================================================================================


class Environment:
    """
    Abstract base class for all AI environments.

    An environment is the "world" where agents exist and interact. It handles:
    1. Managing all objects (agents and things) in the world
    2. Processing agent actions and updating world state
    3. Providing percepts (sensory information) to agents
    4. Running the simulation loop that drives agent-environment interaction

    Key Design Pattern: The environment is responsible for the physics and
    rules of the world, while agents are responsible for intelligence.

    REQUIRED METHODS (must implement in subclasses):
        percept(agent): What can this agent sense right now?
        execute_action(agent, action): How does this action change the world?

    SIMULATION LOOP (Environment.run()):
        1. Get percepts for all agents
        2. Let each agent choose an action based on its percept
        3. Execute all actions and update world state
        4. Apply any spontaneous changes (exogenous_change)
        5. Check if simulation should continue (is_done)
        6. Repeat until done

    For students: Think of Environment as the "game engine" that makes
    the AI simulation possible!
    """

    def __init__(self):
        self.things = []  # All objects in the environment (agents + things)
        self.agents = []  # Active agents (subset of things)

    def thing_classes(self):
        """
        Return list of Thing classes that can exist in this environment.
        Override in subclasses to specify which objects are allowed.
        """
        return []  # Default: empty list means any Thing can be added

    def percept(self, agent):
        """
        Return the percept that the specified agent receives.

        This is the SENSOR function - what can the agent observe about
        its current situation? The percept format depends on the environment.

        Args:
            agent (Agent): The agent requesting sensory information

        Returns:
            Percept: Environment-specific data (location, objects, status, etc.)

        NOTE: Must implement in subclasses! This defines what agents can sense.
        """
        raise NotImplementedError

    def execute_action(self, agent, action):
        """
        Process an agent's action and update the environment state.

        This is the ACTUATOR function - how do agent actions change the world?
        Should also update the agent's performance measure.

        Args:
            agent (Agent): The agent performing the action
            action: The action to execute (format depends on environment)

        NOTE: Must implement in subclasses! This defines how actions work.
        """
        raise NotImplementedError

    def default_location(self, thing):
        """
        Return default location for placing a new thing in the environment.
        Override in subclasses to implement spatial layout logic.
        """
        return None

    def exogenous_change(self):
        """
        Apply spontaneous changes to the environment (if any).

        "Exogenous" = external, not caused by agents
        Examples: weather changes, objects appearing/disappearing,
        non-agent entities moving, etc.

        Called once per simulation step after all agent actions.
        Override in subclasses that need dynamic environments.
        """
        pass

    def is_done(self):
        """
        Check if the simulation should terminate.

        Returns:
            bool: True if simulation should stop, False to continue

        Default: Stop when no agents are alive
        Override to implement custom termination conditions
        (goal achieved, time limit, etc.)
        """
        return not any(agent.is_alive() for agent in self.agents)

    def step(self):
        """
        Execute one time step of the agent-environment interaction loop.

        This is the heart of the AI simulation! Each step:
        1. Check if simulation should continue
        2. Collect actions from all living agents
        3. Execute all actions simultaneously
        4. Apply any exogenous changes

        The parallel execution (collect all actions first, then execute)
        prevents timing advantages and makes the simulation fair.
        """
        if not self.is_done():
            actions = []
            # Phase 1: Deliberation - let each agent choose an action
            for agent in self.agents:
                if agent.alive:
                    # Agent's program processes percept and returns action
                    actions.append(agent.program(self.percept(agent)))
                else:
                    actions.append("")  # Dead agents can't act

            # Phase 2: Action - execute all actions simultaneously
            for agent, action in zip(self.agents, actions):
                self.execute_action(agent, action)

            # Phase 3: World dynamics - spontaneous changes
            self.exogenous_change()

    def run(self, steps=1000):
        """
        Run the complete simulation for a specified number of time steps.

        Args:
            steps (int): Maximum number of simulation steps

        Note: May terminate early if is_done() returns True
        (e.g., all agents dead, goal achieved, etc.)
        """
        for step in range(steps):
            if self.is_done():
                return  # Early termination
            self.step()

    def list_things_at(self, location, tclass=Thing):
        """
        Return all things of specified type at the given location.

        Args:
            location: Position to search (format depends on environment)
            tclass (class): Type of thing to look for (default: any Thing)

        Returns:
            list: All matching things at that location

        Note: Handles both 1D (single number) and multi-D (tuple) locations
        """
        if isinstance(location, numbers.Number):
            # 1D location (simple number)
            return [
                thing
                for thing in self.things
                if thing.location == location and isinstance(thing, tclass)
            ]
        # Multi-dimensional location (tuple/list)
        return [
            thing
            for thing in self.things
            if all(x == y for x, y in zip(thing.location, location))
            and isinstance(thing, tclass)
        ]

    def some_things_at(self, location, tclass=Thing):
        """
        Check if any things of specified type exist at the given location.

        Args:
            location: Position to check
            tclass (class): Type of thing to look for

        Returns:
            bool: True if at least one matching thing found

        More efficient than list_things_at() when you just need to know
        if something exists there (doesn't build the full list).
        """
        return self.list_things_at(location, tclass) != []

    def add_thing(self, thing, location=None):
        """
        Add a new thing to the environment at the specified location.

        Args:
            thing (Thing or function): Object to add, or agent program
            location: Where to place the thing (None = use default_location)

        Note: If 'thing' is actually an agent program (function), this
        automatically wraps it in an Agent object for convenience.

        This is the main way to populate your environment with agents and objects!
        """
        # Convenience feature: auto-wrap agent programs in Agent objects
        if not isinstance(thing, Thing):
            thing = Agent(thing)

        # Prevent duplicate additions
        if thing in self.things:
            print("Can't add the same thing twice")
        else:
            # Set location (or use environment's default)
            thing.location = (
                location if location is not None else self.default_location(thing)
            )
            self.things.append(thing)

            # Special handling for agents
            if isinstance(thing, Agent):
                thing.performance = 0  # Initialize performance tracking
                self.agents.append(thing)  # Add to agent list for simulation loop

    def delete_thing(self, thing):
        """
        Remove a thing from the environment.

        Args:
            thing (Thing): Object to remove

        Handles cleanup of both the main things list and agents list.
        Includes error handling for robustness.
        """
        try:
            self.things.remove(thing)
        except ValueError as e:
            # Helpful debugging information if removal fails
            print(e)
            print("  in Environment delete_thing")
            print("  Thing to be removed: {} at {}".format(thing, thing.location))
            print(
                "  from list: {}".format(
                    [(thing, thing.location) for thing in self.things]
                )
            )

        # Also remove from agents list if applicable
        if thing in self.agents:
            self.agents.remove(thing)


class Direction:
    """
    Directional navigation system for 2D agent movement.

    This utility class helps agents navigate in 2D grid environments by:
    1. Representing cardinal directions (North, South, East, West)
    2. Enabling rotation (turning left/right)
    3. Computing forward movement in the current direction

    Class Constants:
        R = "right" (East, positive X direction)
        L = "left"  (West, negative X direction)
        U = "up"    (North, negative Y direction - screen coordinates!)
        D = "down"  (South, positive Y direction)

    Coordinate System: Uses standard computer graphics coordinates
    where (0,0) is top-left, X increases rightward, Y increases downward.

    Usage Example:
        d = Direction("down")      # Agent facing south
        d = d + "right"           # Turn clockwise (now facing west)
        new_pos = d.move_forward((5,5))  # Move west to (4,5)

    Note: Addition only accepts "right" and "left" for turning!
    """

    # Cardinal direction constants
    R = "right"  # East (+X direction)
    L = "left"  # West (-X direction)
    U = "up"  # North (-Y direction, screen coordinates)
    D = "down"  # South (+Y direction)

    def __init__(self, direction):
        """Initialize with a cardinal direction string."""
        self.direction = direction

    def __add__(self, heading):
        """
        Turn the direction left or right (rotation operation).

        Args:
            heading (str): Either "right" (clockwise) or "left" (counter-clockwise)

        Returns:
            Direction: New direction after turning

        Rotation Logic (clockwise = "right"):
            right → down → left → up → right (cycle)

        Examples:
            >>> d = Direction('right')
            >>> d_turned = d + "right"  # Turn clockwise
            >>> d_turned.direction
            'down'

        Note: This is the only way to change direction! Agents must
        explicitly turn rather than jumping to arbitrary directions.
        """
        if self.direction == self.R:
            return {
                self.R: Direction(self.D),  # right + right = down
                self.L: Direction(self.U),  # right + left = up
            }.get(heading, None)
        elif self.direction == self.L:
            return {
                self.R: Direction(self.U),  # left + right = up
                self.L: Direction(self.D),  # left + left = down
            }.get(heading, None)
        elif self.direction == self.U:
            return {
                self.R: Direction(self.R),  # up + right = right
                self.L: Direction(self.L),  # up + left = left
            }.get(heading, None)
        elif self.direction == self.D:
            return {
                self.R: Direction(self.L),  # down + right = left
                self.L: Direction(self.R),  # down + left = right
            }.get(heading, None)

    def move_forward(self, from_location):
        """
        Calculate the position after moving one step forward in current direction.

        Args:
            from_location: Current position (tuple, list, or other iterable)

        Returns:
            Same type as from_location: New position after moving forward

        Movement Rules (screen/grid coordinates):
            - "right": X increases (move east)
            - "left":  X decreases (move west)
            - "up":    Y decreases (move north - screen coordinates!)
            - "down":  Y increases (move south)

        Examples:
            >>> d = Direction('up')
            >>> new_pos = d.move_forward((0, 0))
            >>> new_pos
            (0, -1)  # Moved north (up on screen)

        Note: Preserves the input type (tuple→tuple, list→list, etc.)
        """
        # Preserve the container type of the input location
        iclass = from_location.__class__
        x, y = from_location

        if self.direction == self.R:
            return iclass((x + 1, y))  # Move east (right)
        elif self.direction == self.L:
            return iclass((x - 1, y))  # Move west (left)
        elif self.direction == self.U:
            return iclass((x, y - 1))  # Move north (up)
        elif self.direction == self.D:
            return iclass((x, y + 1))  # Move south (down)


class XYEnvironment(Environment):
    """
    2D Grid-Based Environment for spatial agent simulations.

    This is a major step up from abstract environments! XYEnvironment provides:
    - 2D coordinate system with (x,y) positions
    - Spatial relationships and movement
    - Collision detection with obstacles
    - Agent inventory management (holding objects)
    - Observer pattern for GUI updates

    Key Features:
        - Discrete or continuous 2D space
        - Bounded regions (with optional walls)
        - Distance-based perception
        - Object interaction (grab/release)
        - Extensible for complex spatial worlds

    Common Actions Supported:
        'TurnRight'/'TurnLeft': Rotate agent direction
        'Forward': Move in current facing direction
        'Grab': Pick up objects at current location
        'Release': Drop carried objects

    Educational Value: This is the foundation for most spatial AI problems
    including robotics, games, and navigation challenges.
    """

    def __init__(self, width=10, height=10):
        super().__init__()

        # Environment dimensions
        self.width = width
        self.height = height

        # Observer pattern for GUI updates
        self.observers = []

        # Coordinate bounds (no walls initially)
        self.x_start, self.y_start = (0, 0)
        self.x_end, self.y_end = (self.width, self.height)

    perceptible_distance = 1  # Default sensor range

    def things_near(self, location, radius=None):
        """
        Find all things within a specified radius of a location.

        Args:
            location (tuple): Center position (x, y)
            radius (float): Search radius (None = use default perceptible_distance)

        Returns:
            list: Tuples of (thing, distance_info) for things within radius

        This implements distance-based perception - agents can sense objects
        within their sensor range. Useful for implementing realistic limitations
        on what agents can observe.
        """
        if radius is None:
            radius = self.perceptible_distance
        radius2 = radius * radius  # Use squared distance for efficiency

        return [
            (thing, radius2 - distance_squared(location, thing.location))
            for thing in self.things
            if distance_squared(location, thing.location) <= radius2
        ]

    def percept(self, agent):
        """
        Default perception: agent senses things within its sensor range.

        Override this in subclasses to implement different perception models
        (e.g., line-of-sight, specific object types, directional sensors).
        """
        return self.things_near(agent.location)

    def execute_action(self, agent, action):
        """
        Process agent actions in 2D space with collision detection.

        Standard 2D Actions:
            'TurnRight': Rotate clockwise (uses Direction class)
            'TurnLeft': Rotate counter-clockwise
            'Forward': Move in current direction (with obstacle checking)
            'Grab': Pick up objects at current location
            'Release': Drop carried objects

        Note: Movement is blocked by Obstacle objects (collision detection).
        The agent's 'bump' flag indicates if last movement was blocked.
        """
        agent.bump = False  # Clear collision flag

        if action == "TurnRight":
            agent.direction += Direction.R
        elif action == "TurnLeft":
            agent.direction += Direction.L
        elif action == "Forward":
            # Attempt movement with collision detection
            new_location = agent.direction.move_forward(agent.location)
            agent.bump = self.move_to(agent, new_location)
        elif action == "Grab":
            # Pick up objects the agent can grab
            things = [
                thing
                for thing in self.list_things_at(agent.location)
                if agent.can_grab(thing)
            ]
            if things:
                agent.holding.append(things[0])
                print("Grabbing ", things[0].__class__.__name__)
                self.delete_thing(things[0])  # Remove from environment
        elif action == "Release":
            # Drop carried objects at current location
            if agent.holding:
                dropped = agent.holding.pop()
                print("Dropping ", dropped.__class__.__name__)
                self.add_thing(dropped, location=agent.location)

    def default_location(self, thing):
        location = self.random_location_inbounds()
        while self.some_things_at(location, Obstacle):
            # we will find a random location with no obstacles
            location = self.random_location_inbounds()
        return location

    def move_to(self, thing, destination):
        """Move a thing to a new location. Returns True on success or False if there is an Obstacle.
        If thing is holding anything, they move with him."""
        thing.bump = self.some_things_at(destination, Obstacle)
        if not thing.bump:
            thing.location = destination
            for o in self.observers:
                o.thing_moved(thing)
            for t in thing.holding:
                self.delete_thing(t)
                self.add_thing(t, destination)
                t.location = destination
        return thing.bump

    def add_thing(self, thing, location=None, exclude_duplicate_class_items=False):
        """Add things to the world. If (exclude_duplicate_class_items) then the item won't be
        added if the location has at least one item of the same class."""
        if location is None:
            super().add_thing(thing)
        elif self.is_inbounds(location):
            if exclude_duplicate_class_items and any(
                isinstance(t, thing.__class__) for t in self.list_things_at(location)
            ):
                return
            super().add_thing(thing, location)

    def is_inbounds(self, location):
        """Checks to make sure that the location is inbounds (within walls if we have walls)"""
        x, y = location
        return not (
            x < self.x_start or x > self.x_end or y < self.y_start or y > self.y_end
        )

    def random_location_inbounds(self, exclude=None):
        """Returns a random location that is inbounds (within walls if we have walls)"""
        location = (
            random.randint(self.x_start, self.x_end),
            random.randint(self.y_start, self.y_end),
        )
        if exclude is not None:
            while location == exclude:
                location = (
                    random.randint(self.x_start, self.x_end),
                    random.randint(self.y_start, self.y_end),
                )
        return location

    def delete_thing(self, thing):
        """Deletes thing, and everything it is holding (if thing is an agent)"""
        if isinstance(thing, Agent):
            del thing.holding

        super().delete_thing(thing)
        for obs in self.observers:
            obs.thing_deleted(thing)

    def add_walls(self):
        """Put walls around the entire perimeter of the grid."""
        for x in range(self.width):
            self.add_thing(Wall(), (x, 0))
            self.add_thing(Wall(), (x, self.height - 1))
        for y in range(1, self.height - 1):
            self.add_thing(Wall(), (0, y))
            self.add_thing(Wall(), (self.width - 1, y))

        # Updates iteration start and end (with walls).
        self.x_start, self.y_start = (1, 1)
        self.x_end, self.y_end = (self.width - 1, self.height - 1)

    def add_observer(self, observer):
        """Adds an observer to the list of observers.
        An observer is typically an EnvGUI.

        Each observer is notified of changes in move_to and add_thing,
        by calling the observer's methods thing_moved(thing)
        and thing_added(thing, loc)."""
        self.observers.append(observer)

    def turn_heading(self, heading, inc):
        """Return the heading to the left (inc=+1) or right (inc=-1) of heading."""
        return turn_heading(heading, inc)


class Obstacle(Thing):
    """
    Immovable barrier that blocks agent movement.

    Obstacles represent physical barriers in the environment - walls, rocks,
    furniture, etc. Agents cannot move into squares containing obstacles,
    and will have their 'bump' flag set to True if they try.

    This enables realistic spatial navigation where agents must plan
    paths around barriers rather than moving freely through solid objects.
    """

    pass


class Wall(Obstacle):
    """
    Specific type of obstacle representing walls/boundaries.

    Commonly used to create bounded environments with perimeter walls.
    See XYEnvironment.add_walls() for automatic wall generation.
    """

    pass


# ================================================================================
# GRAPHICAL ENVIRONMENT: Visual Simulation with Real-Time Display
# ================================================================================


class GraphicEnvironment(XYEnvironment):
    """
    Visual 2D Environment with real-time graphical display.

    Extends XYEnvironment to provide visual feedback using colored grid blocks.
    This is invaluable for:
    - Understanding agent behavior patterns
    - Debugging navigation algorithms
    - Educational demonstrations
    - Creating engaging AI simulations

    Features:
        - Color-coded object representation
        - Real-time animation during simulation
        - Customizable visual themes
        - Performance monitoring capabilities

    Usage for students: Use this whenever you want to SEE what your
    agents are doing! Much easier than reading text logs.
    """

    def __init__(self, width=10, height=10, boundary=True, color={}, display=False):
        """
        Initialize graphical environment with visual grid.

        Args:
            width, height (int): Grid dimensions
            boundary (bool): Whether environment has boundaries
            color (dict): Color mapping {ClassName: (R,G,B)}
            display (bool): Show grid immediately after creation
        """
        super().__init__(width, height)

        # Initialize visual grid (requires ipythonblocks)
        self.grid = BlockGrid(
            width, height, fill=(200, 200, 200)
        )  # Light gray background

        if display:
            self.grid.show()
            self.visible = True
        else:
            self.visible = False

        self.bounded = boundary
        self.colors = color  # Color scheme for different object types

    def get_world(self):
        """
        Generate 2D array representation of current world state.

        Returns:
            list: 2D array where result[x][y] = list of things at position (x,y)

        This converts the environment's internal representation into a format
        suitable for rendering in the BlockGrid visualization.
        """
        result = []
        x_start, y_start = (0, 0)
        x_end, y_end = self.width, self.height

        for x in range(x_start, x_end):
            row = []
            for y in range(y_start, y_end):
                row.append(self.list_things_at((x, y)))
            result.append(row)
        return result

    def run(self, steps=1000, delay=1):
        """
        Run simulation with visual updates at each step.

        Args:
            steps (int): Maximum simulation steps
            delay (float): Pause between steps (seconds) for animation effect

        This overrides the parent run() method to include visual updates,
        creating an animated simulation that shows agent behavior in real-time.
        """
        for step in range(steps):
            self.update(delay)  # Visual update with timing
            if self.is_done():
                break
            self.step()
        self.update(delay)  # Final display update

    def update(self, delay=1):
        """Update the visual display with specified delay."""
        sleep(delay)
        self.reveal()

    def reveal(self):
        """
        Refresh the visual display to show current world state.

        This renders all objects in their current positions using the
        color scheme defined in self.colors. The most recently added
        object at each location determines the cell color.
        """
        self.draw_world()
        # Clear previous output and show updated grid
        clear_output(1)
        self.grid.show()
        self.visible = True

    def draw_world(self):
        """Apply color scheme to grid based on object positions."""
        self.grid[:] = (200, 200, 200)  # Reset to background color
        world = self.get_world()

        for x in range(0, len(world)):
            for y in range(0, len(world[x])):
                if len(world[x][y]):
                    # Use color of topmost object (last in list)
                    obj_class = world[x][y][-1].__class__.__name__
                    if obj_class in self.colors:
                        self.grid[y, x] = self.colors[obj_class]

    def conceal(self):
        """Hide the visual display."""
        self.visible = False
        display(HTML(""))


# ================================================================================
# SPECIALIZED ENVIRONMENTS: Domain-Specific AI Challenge Worlds
# ================================================================================


# Continuous Environment for Advanced Spatial Reasoning
class ContinuousWorld(Environment):
    """
    Continuous (non-grid) spatial environment with polygon obstacles.

    Unlike grid-based environments, this supports:
    - Real-valued coordinates (not just integer grid positions)
    - Complex polygon-shaped obstacles
    - Smooth movement and rotation
    - More realistic physics simulation

    Useful for advanced robotics and navigation research where
    discrete grids are too limiting.
    """

    def __init__(self, width=10, height=10):
        super().__init__()
        self.width = width
        self.height = height

    def add_obstacle(self, coordinates):
        """Add a polygon-shaped obstacle defined by coordinate vertices."""
        self.things.append(PolygonObstacle(coordinates))


class PolygonObstacle(Obstacle):
    """
    Complex geometric obstacle defined by polygon vertices.

    Enables more realistic obstacle shapes than simple grid squares.
    """

    def __init__(self, coordinates):
        """
        Args:
            coordinates (list): List of (x,y) tuples defining polygon vertices
        """
        super().__init__()
        self.coordinates = coordinates


# ================================================================================
# VACUUM WORLD: Classic AI Environment for Performance Measurement
# ================================================================================


class Dirt(Thing):
    """
    Dirt particles that can be cleaned by vacuum agents.

    Simple objects that represent the "mess" that cleaning agents
    are designed to remove. Used in vacuum world environments
    to test agent cleaning strategies.
    """

    pass


class VacuumEnvironment(XYEnvironment):
    """
    [Exercise 2.12] Advanced 2D Vacuum World with Performance Measurement

    This environment tests cleaning agents in a realistic 2D space with:
    - Walls creating boundaries and obstacles
    - Dirt scattered throughout the space
    - Performance scoring: +100 per dirt cleaned, -1 per action
    - Limited perception (agent can't see whole environment)

    Key Differences from TrivialVacuumEnvironment:
    1. 2D space instead of just 2 locations
    2. Walls block movement (collision detection)
    3. Agent doesn't perceive its location directly
    4. More complex navigation required

    Performance Metrics:
        +100 points: Each dirt particle cleaned
        -1 point: Each action taken (movement cost)

    This creates interesting trade-offs between thorough cleaning
    and efficient movement patterns.

    Educational value: Shows how environment complexity affects
    required agent intelligence and strategy.
    """

    def __init__(self, width=10, height=10):
        super().__init__(width, height)
        self.add_walls()  # Create bounded environment

    def thing_classes(self):
        """Objects that can exist in this vacuum world."""
        return [
            Wall,
            Dirt,
            ReflexVacuumAgent,
            RandomVacuumAgent,
            TableDrivenVacuumAgent,
            ModelBasedVacuumAgent,
        ]

    def percept(self, agent):
        """
        Agent receives local cleanliness status and collision info.

        Returns:
            tuple: ('Dirty'/'Clean', 'Bump'/'None')

        Note: Unlike TrivialVacuumEnvironment, location is NOT provided!
        This forces agents to navigate by memory and local sensing only.
        """
        status = "Dirty" if self.some_things_at(agent.location, Dirt) else "Clean"
        bump = "Bump" if agent.bump else "None"
        return status, bump

    def execute_action(self, agent, action):
        """Process cleaning and movement actions with performance tracking."""
        agent.bump = False

        if action == "Suck":
            # Cleaning action - remove dirt and award points
            dirt_list = self.list_things_at(agent.location, Dirt)
            if dirt_list != []:
                dirt = dirt_list[0]
                agent.performance += 100  # Reward for cleaning
                self.delete_thing(dirt)
        else:
            # Movement actions handled by parent class
            super().execute_action(agent, action)

        # All actions cost energy (except NoOp)
        if action != "NoOp":
            agent.performance -= 1


class TrivialVacuumEnvironment(Environment):
    """
    Simple Two-Location Vacuum World for Educational Demonstrations

    This is the classic introductory AI environment from the textbook!
    Features just two locations (A and B) that can each be Clean or Dirty.

    Perfect for:
    - Understanding basic agent-environment interaction
    - Comparing different agent architectures
    - Learning performance measurement concepts
    - Testing agent programs without spatial complexity

    Environment Properties:
        - Fully observable (agent knows its location)
        - Deterministic (actions have predictable effects)
        - Static (no changes unless agent acts)
        - Discrete (finite states and actions)

    Percept Format: (location, status)
        - location: loc_A (0,0) or loc_B (1,0)
        - status: 'Clean' or 'Dirty'

    Actions: 'Left', 'Right', 'Suck'

    Performance Measure:
        +10 points: Each dirt cleaned
        -1 point: Each movement action

    This creates a simple optimization problem: clean both locations
    with minimal movement.
    """

    def __init__(self):
        super().__init__()
        # Randomly initialize dirt status for interesting scenarios
        self.status = {
            loc_A: random.choice(["Clean", "Dirty"]),
            loc_B: random.choice(["Clean", "Dirty"]),
        }

    def thing_classes(self):
        """Objects allowed in this simple environment."""
        return [
            Wall,
            Dirt,
            ReflexVacuumAgent,
            RandomVacuumAgent,
            TableDrivenVacuumAgent,
            ModelBasedVacuumAgent,
        ]

    def percept(self, agent):
        """
        Agent perceives its exact location and local cleanliness status.

        Returns:
            tuple: (location, status) where location is loc_A or loc_B
        """
        return agent.location, self.status[agent.location]

    def execute_action(self, agent, action):
        """
        Process agent actions and update performance score.

        Actions:
            'Right': Move from A to B (-1 point)
            'Left': Move from B to A (-1 point)
            'Suck': Clean current location (+10 if dirty, 0 if clean)
        """
        if action == "Right":
            agent.location = loc_B
            agent.performance -= 1  # Movement cost
        elif action == "Left":
            agent.location = loc_A
            agent.performance -= 1  # Movement cost
        elif action == "Suck":
            if self.status[agent.location] == "Dirty":
                agent.performance += 10  # Cleaning reward
            self.status[agent.location] = "Clean"  # Always becomes clean

    def default_location(self, thing):
        """Place new agents randomly at location A or B."""
        return random.choice([loc_A, loc_B])


# ================================================================================
# THE WUMPUS WORLD: Classic AI Challenge Environment
# ================================================================================

# Game Objects and Percepts


class Gold(Thing):
    """
    Treasure object that the Explorer seeks in Wumpus World.

    The goal of the Wumpus World game is to find the gold and return
    safely to the entrance. Gold produces a 'Glitter' percept when
    the agent is in the same square.
    """

    def __eq__(self, rhs):
        """All Gold objects are considered equal (for game logic)."""
        return rhs.__class__ == Gold

    pass


class Bump(Thing):
    """Percept indicating the agent tried to move into a wall."""

    pass


class Glitter(Thing):
    """Percept indicating gold is present in the current square."""

    pass


class Pit(Thing):
    """
    Deadly trap that kills the Explorer if entered.

    Pits are randomly distributed throughout the cave (except entrance).
    They produce 'Breeze' percepts in adjacent squares as a warning.
    """

    pass


class Breeze(Thing):
    """Percept indicating a pit is in an adjacent square."""

    pass


class Arrow(Thing):
    """Projectile that can kill the Wumpus (Explorer starts with one)."""

    pass


class Scream(Thing):
    """Percept produced when the Wumpus is killed by an arrow."""

    pass


class Wumpus(Agent):
    """
    The monster of Wumpus World - deadly and smelly!

    The Wumpus is a stationary creature that:
    - Kills the Explorer if they enter the same square
    - Produces 'Stench' percepts in adjacent squares
    - Can be killed by the Explorer's arrow
    - Lets out a 'Scream' when killed
    """

    screamed = False  # Track if death scream has been emitted
    pass


class Stench(Thing):
    """Percept indicating the Wumpus is in an adjacent square."""

    pass


class Explorer(Agent):
    """
    The player agent in Wumpus World.

    The Explorer's mission: navigate the dangerous cave, find the gold,
    and return safely to the entrance at (1,1).

    Equipment and Abilities:
        - One arrow (can kill Wumpus)
        - Can grab gold
        - Directional movement
        - Remembers what killed them (for learning)
    """

    holding = []  # Objects currently carried
    has_arrow = True  # Starts with one arrow
    killed_by = ""  # What killed the agent (for analysis)
    direction = Direction("right")  # Initially facing east

    def can_grab(self, thing):
        """Explorer can only grab gold (not pits, walls, etc.)."""
        return thing.__class__ == Gold


class WumpusEnvironment(XYEnvironment):
    """
    The Famous Wumpus World: A Classic AI Reasoning Challenge

    This environment implements the complete Wumpus World from Chapter 7
    of AIMA, featuring logical reasoning under uncertainty, risk assessment,
    and goal-directed planning.

    WORLD SETUP:
        - 4x4 grid of rooms (plus walls = 6x6 total)
        - One Wumpus (randomly placed, not at entrance)
        - One Gold (randomly placed, not at entrance)
        - Pits (20% probability per room, not at entrance)
        - Explorer starts at (1,1) with one arrow

    PERCEPT SYSTEM (Indirect Evidence):
        - Stench: Wumpus in adjacent room
        - Breeze: Pit in adjacent room
        - Glitter: Gold in current room
        - Bump: Tried to move into wall
        - Scream: Wumpus was killed

    ACTIONS:
        - Forward, TurnLeft, TurnRight: Movement
        - Grab: Pick up gold
        - Shoot: Fire arrow in facing direction
        - Climb: Exit cave (only at entrance)

    PERFORMANCE MEASURE:
        +1000: Climb out with gold
        -1000: Death by pit or Wumpus
        -1: Each action taken (efficiency matters!)

    KEY AI CONCEPTS DEMONSTRATED:
        - Logical inference from indirect evidence
        - Risk assessment and decision theory
        - Knowledge representation and reasoning
        - Planning under uncertainty
        - Propositional logic and constraint satisfaction

    For students: This is where simple reflex agents fail and you need
    sophisticated reasoning capabilities!
    """

    pit_probability = 0.2  # 20% chance of pit per room (from Chapter 7.2)

    def __init__(self, agent_program, width=6, height=6):
        """
        Initialize Wumpus World with standard 4x4 room layout.

        Args:
            agent_program: The Explorer's decision-making function
            width, height: Environment dimensions (6x6 = 4x4 rooms + walls)
        """
        super().__init__(width, height)
        self.init_world(agent_program)

    def init_world(self, program):
        """
        Generate the Wumpus World according to standard probabilities.

        World Generation Rules:
        1. Surround with walls (creates bounded 4x4 room space)
        2. Place pits randomly (20% per room, except entrance)
        3. Place Wumpus randomly (not at entrance)
        4. Place gold randomly (not at entrance)
        5. Add Explorer at entrance (1,1)
        6. Generate appropriate percepts (stenches, breezes)
        """

        # WALLS: Create perimeter boundary
        self.add_walls()

        # PITS: Random placement with adjacency warnings
        for x in range(self.x_start, self.x_end):
            for y in range(self.y_start, self.y_end):
                # Skip entrance square - must be safe!
                if (x, y) != (1, 1) and random.random() < self.pit_probability:
                    self.add_thing(Pit(), (x, y), True)
                    # Add breeze warnings in adjacent squares
                    self.add_thing(Breeze(), (x - 1, y), True)
                    self.add_thing(Breeze(), (x, y - 1), True)
                    self.add_thing(Breeze(), (x + 1, y), True)
                    self.add_thing(Breeze(), (x, y + 1), True)

        # WUMPUS: Single monster placement with stench warnings
        w_x, w_y = self.random_location_inbounds(exclude=(1, 1))
        self.add_thing(Wumpus(lambda x: ""), (w_x, w_y), True)
        # Add stench warnings in adjacent squares
        self.add_thing(Stench(), (w_x - 1, w_y), True)
        self.add_thing(Stench(), (w_x + 1, w_y), True)
        self.add_thing(Stench(), (w_x, w_y - 1), True)
        self.add_thing(Stench(), (w_x, w_y + 1), True)

        # GOLD: Treasure placement (goal object)
        self.add_thing(Gold(), self.random_location_inbounds(exclude=(1, 1)), True)

        # EXPLORER: Player agent at safe entrance
        self.add_thing(Explorer(program), (1, 1), True)

    def get_world(self, show_walls=True):
        """Return the items in the world"""
        result = []
        x_start, y_start = (0, 0) if show_walls else (1, 1)

        if show_walls:
            x_end, y_end = self.width, self.height
        else:
            x_end, y_end = self.width - 1, self.height - 1

        for x in range(x_start, x_end):
            row = []
            for y in range(y_start, y_end):
                row.append(self.list_things_at((x, y)))
            result.append(row)
        return result

    def percepts_from(self, agent, location, tclass=Thing):
        """Return percepts from a given location,
        and replaces some items with percepts from chapter 7."""
        thing_percepts = {
            Gold: Glitter(),
            Wall: Bump(),
            Wumpus: Stench(),
            Pit: Breeze(),
        }

        """Agents don't need to get their percepts"""
        thing_percepts[agent.__class__] = None

        """Gold only glitters in its cell"""
        if location != agent.location:
            thing_percepts[Gold] = None

        result = [
            thing_percepts.get(thing.__class__, thing)
            for thing in self.things
            if thing.location == location and isinstance(thing, tclass)
        ]
        return result if len(result) else [None]

    def percept(self, agent):
        """Return things in adjacent (not diagonal) cells of the agent.
        Result format: [Left, Right, Up, Down, Center / Current location]"""
        x, y = agent.location
        result = []
        result.append(self.percepts_from(agent, (x - 1, y)))
        result.append(self.percepts_from(agent, (x + 1, y)))
        result.append(self.percepts_from(agent, (x, y - 1)))
        result.append(self.percepts_from(agent, (x, y + 1)))
        result.append(self.percepts_from(agent, (x, y)))

        """The wumpus gives out a loud scream once it's killed."""
        wumpus = [thing for thing in self.things if isinstance(thing, Wumpus)]
        if len(wumpus) and not wumpus[0].alive and not wumpus[0].screamed:
            result[-1].append(Scream())
            wumpus[0].screamed = True

        return result

    def execute_action(self, agent, action):
        """Modify the state of the environment based on the agent's actions.
        Performance score taken directly out of the book."""

        if isinstance(agent, Explorer) and self.in_danger(agent):
            return

        agent.bump = False
        if action in ["TurnRight", "TurnLeft", "Forward", "Grab"]:
            super().execute_action(agent, action)
            agent.performance -= 1
        elif action == "Climb":
            if agent.location == (1, 1):  # Agent can only climb out of (1,1)
                agent.performance += 1000 if Gold() in agent.holding else 0
                self.delete_thing(agent)
        elif action == "Shoot":
            """The arrow travels straight down the path the agent is facing"""
            if agent.has_arrow:
                arrow_travel = agent.direction.move_forward(agent.location)
                while self.is_inbounds(arrow_travel):
                    wumpus = [
                        thing
                        for thing in self.list_things_at(arrow_travel)
                        if isinstance(thing, Wumpus)
                    ]
                    if len(wumpus):
                        wumpus[0].alive = False
                        break
                    arrow_travel = agent.direction.move_forward(agent.location)
                agent.has_arrow = False

    def in_danger(self, agent):
        """Check if Explorer is in danger (Pit or Wumpus), if he is, kill him"""
        for thing in self.list_things_at(agent.location):
            if isinstance(thing, Pit) or (isinstance(thing, Wumpus) and thing.alive):
                agent.alive = False
                agent.performance -= 1000
                agent.killed_by = thing.__class__.__name__
                return True
        return False

    def is_done(self):
        """The game is over when the Explorer is killed
        or if he climbs out of the cave only at (1,1)."""
        explorer = [agent for agent in self.agents if isinstance(agent, Explorer)]
        if len(explorer):
            if explorer[0].alive:
                return False
            else:
                print("Death by {} [-1000].".format(explorer[0].killed_by))
        else:
            print(
                "Explorer climbed out {}.".format(
                    "with Gold [+1000]!"
                    if Gold() not in self.things
                    else "without Gold [+0]"
                )
            )
        return True

    # TODO: Complete arrow shooting mechanics implementation


# ================================================================================
# AGENT PERFORMANCE EVALUATION AND TESTING UTILITIES
# ================================================================================


def compare_agents(EnvFactory, AgentFactories, n=10, steps=1000):
    """
    Scientific comparison of multiple agent types across multiple trials.

    This function runs controlled experiments to evaluate agent performance,
    providing statistical evidence for which agents work better in specific
    environments. Essential for AI research and agent development!

    Args:
        EnvFactory (callable): Constructor for environment instances
        AgentFactories (list): List of agent constructors to test
        n (int): Number of trial runs per agent (for statistical significance)
        steps (int): Maximum steps per trial

    Returns:
        list: Tuples of (AgentClass, average_performance) sorted by performance

    Example:
        >>> env_factory = TrivialVacuumEnvironment
        >>> agents = [ModelBasedVacuumAgent, ReflexVacuumAgent, RandomVacuumAgent]
        >>> results = compare_agents(env_factory, agents, n=20)
        >>> for agent_class, avg_score in results:
        ...     print(f"{agent_class.__name__}: {avg_score:.2f}")

    Statistical Note: Multiple trials (n > 1) account for random environment
    variations and provide confidence in performance differences.
    """
    # Create n independent environment instances for fair testing
    envs = [EnvFactory() for i in range(n)]

    return [(A, test_agent(A, steps, copy.deepcopy(envs))) for A in AgentFactories]


def test_agent(AgentFactory, steps, envs):
    """
    Test a single agent type across multiple environment instances.

    Args:
        AgentFactory (callable): Constructor for the agent type to test
        steps (int): Maximum steps per environment
        envs (list): List of environment instances to test on

    Returns:
        float: Mean performance score across all test environments

    This function isolates the testing of one agent type, ensuring
    fair comparison by using identical environment setups for all agents.
    """

    def score(env):
        """Run one trial and return the agent's final performance."""
        agent = AgentFactory()
        env.add_thing(agent)
        env.run(steps)
        return agent.performance

    return mean(map(score, envs))


# ================================================================================
# EXAMPLE USAGE AND DOCTESTS
# ================================================================================

# The following examples demonstrate basic agent-environment interaction
# and serve as automated tests for the framework functionality.


__doc__ += """
>>> a = ReflexVacuumAgent()
>>> a.program((loc_A, 'Clean'))
'Right'
>>> a.program((loc_B, 'Clean'))
'Left'
>>> a.program((loc_A, 'Dirty'))
'Suck'
>>> a.program((loc_A, 'Dirty'))
'Suck'

>>> e = TrivialVacuumEnvironment()
>>> e.add_thing(ModelBasedVacuumAgent())
>>> e.run(5)

"""
