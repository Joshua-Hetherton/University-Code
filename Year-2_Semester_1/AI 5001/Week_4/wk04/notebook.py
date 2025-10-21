import time
from collections import defaultdict
from inspect import getsource

import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import HTML
from IPython.display import display
from PIL import Image
from matplotlib import lines

from logic import parse_definite_clause, standardize_variables, unify_mm, subst
from search import GraphProblem, romania_map


# ______________________________________________________________________________
# Magic Words - Utility functions for displaying algorithm pseudocode and source code


def pseudocode(algorithm):
    """Print the pseudocode for the given algorithm.

    Fetches algorithm pseudocode from the AIMA pseudocode repository online
    and displays it as formatted Markdown in Jupyter notebooks.

    Args:
        algorithm (str): Name of the algorithm (spaces will be converted to hyphens)

    Returns:
        IPython.display.Markdown: Formatted pseudocode for display
    """
    from urllib.request import urlopen
    from IPython.display import Markdown

    algorithm = algorithm.replace(" ", "-")
    url = "https://raw.githubusercontent.com/aimacode/aima-pseudocode/master/md/{}.md".format(
        algorithm
    )
    f = urlopen(url)
    md = f.read().decode("utf-8")
    md = md.split("\n", 1)[-1].strip()
    md = "#" + md
    return Markdown(md)


def psource(*functions):
    """Print the source code for the given function(s).

    Displays Python source code with syntax highlighting using Pygments.
    Falls back to plain text if Pygments is not available.

    Args:
        *functions: Variable number of function objects to display

    Note:
        Uses Pygments for syntax highlighting when available,
        otherwise displays plain source code.
    """
    source_code = "\n\n".join(getsource(fn) for fn in functions)
    try:
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer
        from pygments import highlight

        display(HTML(highlight(source_code, PythonLexer(), HtmlFormatter(full=True))))

    except ImportError:
        print(source_code)


# ______________________________________________________________________________
# MDP (Markov Decision Process) Visualization - Interactive grid world plotting


def make_plot_grid_step_function(columns, rows, U_over_time):
    """Create an interactive plotting function for MDP grid visualization.

    This higher-order function returns a plotting function compatible with
    ipywidgets interactive displays. Used for visualizing value iteration
    and policy iteration algorithms step-by-step.

    Args:
        columns (int): Number of grid columns
        rows (int): Number of grid rows
        U_over_time (list): List of utility dictionaries over iterations,
                           where each dict maps (x,y) coordinates to utility values

    Returns:
        function: Interactive plotting function that takes iteration number as input

    Note:
        The returned function displays grid with utility values as text overlays.
        Uses blue-white-red colormap where blue=negative, white=zero, red=positive.
    """

    def plot_grid_step(iteration):
        """Plot grid world state for a specific iteration.

        Args:
            iteration (int): Which iteration to display from U_over_time
        """
        # Get utility values for this iteration
        data = U_over_time[iteration]
        data = defaultdict(lambda: 0, data)  # Default to 0 for missing coordinates

        # Build 2D grid from coordinate-value pairs
        grid = []
        for row in range(rows):
            current_row = []
            for column in range(columns):
                current_row.append(data[(column, row)])
            grid.append(current_row)
        grid.reverse()  # Flip vertically to match book convention (origin at bottom-left)

        # Create heatmap visualization
        fig = plt.imshow(grid, cmap=plt.cm.bwr, interpolation="nearest")

        # Hide axes for cleaner appearance
        plt.axis("off")
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        # Add utility values as text overlays on each grid cell
        for col in range(len(grid)):
            for row in range(len(grid[0])):
                utility_value = grid[col][row]
                fig.axes.text(
                    row, col, "{0:.2f}".format(utility_value), va="center", ha="center"
                )

        plt.show()

    return plot_grid_step


def make_visualize(slider):
    """Create a callback function for animated visualization control.

    Returns a function that can animate through slider values automatically.
    Used with ipywidgets to provide play/pause functionality for algorithm
    visualizations.

    Args:
        slider (ipywidgets.IntSlider): The slider widget to animate

    Returns:
        function: Callback function for visualization control that takes
                 (visualize_bool, time_step) as parameters

    Note:
        When visualize=True, automatically steps through slider.min to slider.max
        with specified time delay between steps.
    """

    def visualize_callback(visualize, time_step):
        """Animate through slider values when visualization is enabled.

        Args:
            visualize (bool): Whether to start animation
            time_step (float): Delay in seconds between animation steps
        """
        if visualize is True:
            # Step through all slider values with time delay
            for i in range(slider.min, slider.max + 1):
                slider.value = i
                time.sleep(float(time_step))

    return visualize_callback


# ______________________________________________________________________________
# HTML Canvas Framework - Interactive canvas widgets for Jupyter notebooks


# HTML template for creating canvas elements with JavaScript integration
_canvas = """
<script type="text/javascript" src="./js/canvas.js"></script>
<div>
<canvas id="{0}" width="{1}" height="{2}" style="background:rgba(158, 167, 184, 0.2);" onclick='click_callback(this, event, "{3}")'></canvas>
</div>

<script> var {0}_canvas_object = new Canvas("{0}");</script>
"""  # noqa: E501 - Long line for HTML template


class Canvas:
    """Base class for managing HTML canvas elements in Jupyter notebooks.

    Provides a Python interface to HTML5 Canvas with JavaScript integration.
    Supports drawing primitives, mouse interaction, and animation. All drawing
    commands are queued and executed via JavaScript when update() is called.

    Args:
        varname (str): Variable name for the canvas object (must match Python variable)
        width (int): Canvas width in pixels (default: 800)
        height (int): Canvas height in pixels (default: 600)
        cid (str): Custom canvas ID, defaults to varname if not provided

    Note:
        To create a canvas: my_canvas = Canvas("my_canvas")
        The variable name must match the string passed to constructor.
        Supports both absolute and normalized coordinate systems.
    """

    def __init__(self, varname, width=800, height=600, cid=None):
        self.name = varname
        self.cid = cid or varname
        self.width = width
        self.height = height
        self.html = _canvas.format(self.cid, self.width, self.height, self.name)
        self.exec_list = []
        display_html(self.html)

    def mouse_click(self, x, y):
        """Override this method to handle mouse click at position (x, y)"""
        raise NotImplementedError

    def mouse_move(self, x, y):
        raise NotImplementedError

    def execute(self, exec_str):
        """Queue a JavaScript command for later execution.

        Commands are stored in exec_list and executed when update() is called.
        This batching approach improves performance by reducing DOM updates.

        Args:
            exec_str (str): JavaScript command to execute (without object prefix)

        Note:
            Commands are automatically prefixed with canvas object name.
            Invalid commands trigger an alert to the user.
        """
        if not isinstance(exec_str, str):
            print("Invalid execution argument:", exec_str)
            self.alert("Received invalid execution command format")
        prefix = "{0}_canvas_object.".format(self.cid)
        self.exec_list.append(prefix + exec_str + ";")

    def fill(self, r, g, b):
        """Changes the fill color to a color in rgb format"""
        self.execute("fill({0}, {1}, {2})".format(r, g, b))

    def stroke(self, r, g, b):
        """Changes the colors of line/strokes to rgb"""
        self.execute("stroke({0}, {1}, {2})".format(r, g, b))

    def strokeWidth(self, w):
        """Changes the width of lines/strokes to 'w' pixels"""
        self.execute("strokeWidth({0})".format(w))

    def rect(self, x, y, w, h):
        """Draw a rectangle with 'w' width, 'h' height and (x, y) as the top-left corner"""
        self.execute("rect({0}, {1}, {2}, {3})".format(x, y, w, h))

    def rect_n(self, xn, yn, wn, hn):
        """Similar to rect(), but the dimensions are normalized to fall between 0 and 1"""
        x = round(xn * self.width)
        y = round(yn * self.height)
        w = round(wn * self.width)
        h = round(hn * self.height)
        self.rect(x, y, w, h)

    def line(self, x1, y1, x2, y2):
        """Draw a line from (x1, y1) to (x2, y2)"""
        self.execute("line({0}, {1}, {2}, {3})".format(x1, y1, x2, y2))

    def line_n(self, x1n, y1n, x2n, y2n):
        """Similar to line(), but the dimensions are normalized to fall between 0 and 1"""
        x1 = round(x1n * self.width)
        y1 = round(y1n * self.height)
        x2 = round(x2n * self.width)
        y2 = round(y2n * self.height)
        self.line(x1, y1, x2, y2)

    def arc(self, x, y, r, start, stop):
        """Draw an arc with (x, y) as centre, 'r' as radius from angles 'start' to 'stop'"""
        self.execute("arc({0}, {1}, {2}, {3}, {4})".format(x, y, r, start, stop))

    def arc_n(self, xn, yn, rn, start, stop):
        """Draw an arc using normalized coordinates (0-1 range).

        Similar to arc(), but coordinates and radius are normalized.
        Useful for resolution-independent drawing.

        Args:
            xn, yn (float): Normalized center coordinates (0.0-1.0)
            rn (float): Normalized radius (0.0-1.0)
            start, stop (float): Start and end angles in degrees

        Note:
            Radius is normalized using the smaller of width/height to maintain
            circular appearance regardless of canvas aspect ratio.
        """
        x = round(xn * self.width)
        y = round(yn * self.height)
        r = round(rn * min(self.width, self.height))
        self.arc(x, y, r, start, stop)

    def clear(self):
        """Clear the HTML canvas"""
        self.execute("clear()")

    def font(self, font):
        """Changes the font of text"""
        self.execute('font("{0}")'.format(font))

    def text(self, txt, x, y, fill=True):
        """Display a text at (x, y)"""
        if fill:
            self.execute('fill_text("{0}", {1}, {2})'.format(txt, x, y))
        else:
            self.execute('stroke_text("{0}", {1}, {2})'.format(txt, x, y))

    def text_n(self, txt, xn, yn, fill=True):
        """Similar to text(), but with normalized coordinates"""
        x = round(xn * self.width)
        y = round(yn * self.height)
        self.text(txt, x, y, fill)

    def alert(self, message):
        """Immediately display an alert"""
        display_html('<script>alert("{0}")</script>'.format(message))

    def update(self):
        """Execute all queued JavaScript commands and clear the queue.

        Sends accumulated drawing commands to the browser for rendering.
        Must be called after drawing operations to make changes visible.
        Automatically clears the command queue after execution.

        Note:
            This batching approach improves performance by minimizing
            JavaScript execution calls and DOM updates.
        """
        exec_code = "<script>\n" + "\n".join(self.exec_list) + "\n</script>"
        self.exec_list = []
        display_html(exec_code)


def display_html(html_string):
    display(HTML(html_string))


################################################################################
# Specialized Canvas Applications - Interactive game and algorithm visualizations


class Canvas_fol_bc_ask(Canvas):
    """fol_bc_ask() on HTML canvas"""

    def __init__(self, varname, kb, query, width=800, height=600, cid=None):
        super().__init__(varname, width, height, cid)
        self.kb = kb
        self.query = query
        self.l = 1 / 20
        self.b = 3 * self.l
        bc_out = list(self.fol_bc_ask())
        if len(bc_out) == 0:
            self.valid = False
        else:
            self.valid = True
            graph = bc_out[0][0][0]
            s = bc_out[0][1]
            while True:
                new_graph = subst(s, graph)
                if graph == new_graph:
                    break
                graph = new_graph
            self.make_table(graph)
        self.context = None
        self.draw_table()

    def fol_bc_ask(self):
        KB = self.kb
        query = self.query

        def fol_bc_or(KB, goal, theta):
            for rule in KB.fetch_rules_for_goal(goal):
                lhs, rhs = parse_definite_clause(standardize_variables(rule))
                for theta1 in fol_bc_and(KB, lhs, unify_mm(rhs, goal, theta)):
                    yield ([(goal, theta1[0])], theta1[1])

        def fol_bc_and(KB, goals, theta):
            if theta is None:
                pass
            elif not goals:
                yield ([], theta)
            else:
                first, rest = goals[0], goals[1:]
                for theta1 in fol_bc_or(KB, subst(theta, first), theta):
                    for theta2 in fol_bc_and(KB, rest, theta1[1]):
                        yield (theta1[0] + theta2[0], theta2[1])

        return fol_bc_or(KB, query, {})

    def make_table(self, graph):
        table = []
        pos = {}
        links = set()
        edges = set()

        def dfs(node, depth):
            if len(table) <= depth:
                table.append([])
            pos = len(table[depth])
            table[depth].append(node[0])
            for child in node[1]:
                child_id = dfs(child, depth + 1)
                links.add(((depth, pos), child_id))
            return (depth, pos)

        dfs(graph, 0)
        y_off = 0.85 / len(table)
        for i, row in enumerate(table):
            x_off = 0.95 / len(row)
            for j, node in enumerate(row):
                pos[(i, j)] = (
                    0.025 + j * x_off + (x_off - self.b) / 2,
                    0.025 + i * y_off + (y_off - self.l) / 2,
                )
        for p, c in links:
            x1, y1 = pos[p]
            x2, y2 = pos[c]
            edges.add((x1 + self.b / 2, y1 + self.l, x2 + self.b / 2, y2))

        self.table = table
        self.pos = pos
        self.edges = edges

    def mouse_click(self, x, y):
        x, y = x / self.width, y / self.height
        for node in self.pos:
            xs, ys = self.pos[node]
            xe, ye = xs + self.b, ys + self.l
            if xs <= x <= xe and ys <= y <= ye:
                self.context = node
                break
        self.draw_table()

    def draw_table(self):
        self.clear()
        self.strokeWidth(3)
        self.stroke(0, 0, 0)
        self.font("12px Arial")
        if self.valid:
            # draw nodes
            for i, j in self.pos:
                x, y = self.pos[(i, j)]
                self.fill(200, 200, 200)
                self.rect_n(x, y, self.b, self.l)
                self.line_n(x, y, x + self.b, y)
                self.line_n(x, y, x, y + self.l)
                self.line_n(x + self.b, y, x + self.b, y + self.l)
                self.line_n(x, y + self.l, x + self.b, y + self.l)
                self.fill(0, 0, 0)
                self.text_n(self.table[i][j], x + 0.01, y + self.l - 0.01)
            # draw edges
            for x1, y1, x2, y2 in self.edges:
                self.line_n(x1, y1, x2, y2)
        else:
            self.fill(255, 0, 0)
            self.rect_n(0, 0, 1, 1)
        # text area
        self.fill(255, 255, 255)
        self.rect_n(0, 0.9, 1, 0.1)
        self.strokeWidth(5)
        self.stroke(0, 0, 0)
        self.line_n(0, 0.9, 1, 0.9)
        self.font("22px Arial")
        self.fill(0, 0, 0)
        self.text_n(
            self.table[self.context[0]][self.context[1]]
            if self.context
            else "Click for text",
            0.025,
            0.975,
        )
        self.update()


############################################################################################################
#####################        Search Algorithm Visualization Functions         ####################
############################################################################################################


def show_map(graph_data, node_colors=None):
    """Display a graph visualization with colored nodes for search algorithm states.

    Creates a visual representation of search problems like the Romania map,
    showing node states during algorithm execution. Used primarily for
    demonstrating search algorithms in educational contexts.

    Args:
        graph_data (dict): Graph data containing:
            - 'graph_dict': Adjacency list representation
            - 'node_colors': Default node colors
            - 'node_positions': (x,y) coordinates for each node
            - 'node_label_positions': Label placement coordinates
            - 'edge_weights': Dictionary of edge costs
        node_colors (dict, optional): Override colors for nodes

    Note:
        Color coding: white=unexplored, orange=frontier, red=current,
        gray=explored, green=solution path. Includes legend for clarity.
    """
    G = nx.Graph(graph_data["graph_dict"])
    node_colors = node_colors or graph_data["node_colors"]
    node_positions = graph_data["node_positions"]
    node_label_pos = graph_data["node_label_positions"]
    edge_weights = graph_data["edge_weights"]

    # Create large figure for clear visualization
    plt.figure(figsize=(18, 13))

    # Draw the complete graph with node colors indicating search states
    nx.draw(
        G,
        pos={k: node_positions[k] for k in G.nodes()},
        node_color=[node_colors[node] for node in G.nodes()],
        linewidths=0.3,
        edgecolors="k",
    )

    # Add node labels with white background for readability
    node_label_handles = nx.draw_networkx_labels(G, pos=node_label_pos, font_size=14)
    [
        label.set_bbox(dict(facecolor="white", edgecolor="none"))
        for label in node_label_handles.values()
    ]

    # Display edge weights (distances/costs) for informed search algorithms
    nx.draw_networkx_edge_labels(
        G, pos=node_positions, edge_labels=edge_weights, font_size=14
    )

    # Create legend explaining node color meanings
    white_circle = lines.Line2D(
        [], [], color="white", marker="o", markersize=15, markerfacecolor="white"
    )
    orange_circle = lines.Line2D(
        [], [], color="orange", marker="o", markersize=15, markerfacecolor="orange"
    )
    red_circle = lines.Line2D(
        [], [], color="red", marker="o", markersize=15, markerfacecolor="red"
    )
    gray_circle = lines.Line2D(
        [], [], color="gray", marker="o", markersize=15, markerfacecolor="gray"
    )
    green_circle = lines.Line2D(
        [], [], color="green", marker="o", markersize=15, markerfacecolor="green"
    )
    plt.legend(
        (white_circle, orange_circle, red_circle, gray_circle, green_circle),
        (
            "Un-explored",
            "Frontier",
            "Currently Exploring",
            "Explored",
            "Final Solution",
        ),
        numpoints=1,
        prop={"size": 16},
        loc=(0.8, 0.75),
    )

    plt.show()


# Helper functions for search visualizations


def final_path_colors(initial_node_colors, problem, solution):
    """Return node colors highlighting the final solution path.

    Creates a color mapping that shows the solution path in green
    while preserving other node colors from the search process.

    Args:
        initial_node_colors (dict): Original node color mapping
        problem: Search problem instance with initial state
        solution (list): Sequence of actions forming the solution path

    Returns:
        dict: Updated color mapping with solution path highlighted in green

    Note:
        Both the starting node and all nodes in the solution path
        are colored green to clearly show the complete solution.
    """
    # Copy original colors to avoid modifying the input
    final_colors = dict(initial_node_colors)
    # Highlight starting position and solution path
    final_colors[problem.initial] = "green"
    for node in solution:
        final_colors[node] = "green"
    return final_colors


def display_visual(graph_data, user_input, algorithm=None, problem=None):
    initial_node_colors = graph_data["node_colors"]
    if user_input is False:

        def slider_callback(iteration):
            # don't show graph for the first time running the cell calling this function
            try:
                show_map(graph_data, node_colors=all_node_colors[iteration])
            except:
                pass

        def visualize_callback(visualize):
            if visualize is True:
                button.value = False

                global all_node_colors

                iterations, all_node_colors, node = algorithm(problem)
                solution = node.solution()
                all_node_colors.append(
                    final_path_colors(all_node_colors[0], problem, solution)
                )

                slider.max = len(all_node_colors) - 1

                for i in range(slider.max + 1):
                    slider.value = i
                    # time.sleep(.5)

        slider = widgets.IntSlider(min=0, max=1, step=1, value=0)
        slider_visual = widgets.interactive(slider_callback, iteration=slider)
        display(slider_visual)

        button = widgets.ToggleButton(value=False)
        button_visual = widgets.interactive(visualize_callback, visualize=button)
        display(button_visual)

    if user_input is True:
        node_colors = dict(initial_node_colors)
        if isinstance(algorithm, dict):
            assert set(algorithm.keys()).issubset(
                {
                    "Breadth First Tree Search",
                    "Depth First Tree Search",
                    "Breadth First Search",
                    "Depth First Graph Search",
                    "Best First Graph Search",
                    "Uniform Cost Search",
                    "Depth Limited Search",
                    "Iterative Deepening Search",
                    "Greedy Best First Search",
                    "A-star Search",
                    "Recursive Best First Search",
                }
            )

            algo_dropdown = widgets.Dropdown(
                description="Search algorithm: ",
                options=sorted(list(algorithm.keys())),
                value="Breadth First Tree Search",
            )
            display(algo_dropdown)
        elif algorithm is None:
            print("No algorithm to run.")
            return 0

        def slider_callback(iteration):
            # don't show graph for the first time running the cell calling this function
            try:
                show_map(graph_data, node_colors=all_node_colors[iteration])
            except:
                pass

        def visualize_callback(visualize):
            if visualize is True:
                button.value = False

                problem = GraphProblem(
                    start_dropdown.value, end_dropdown.value, romania_map
                )
                global all_node_colors

                user_algorithm = algorithm[algo_dropdown.value]

                iterations, all_node_colors, node = user_algorithm(problem)
                solution = node.solution()
                all_node_colors.append(
                    final_path_colors(all_node_colors[0], problem, solution)
                )

                slider.max = len(all_node_colors) - 1

                for i in range(slider.max + 1):
                    slider.value = i
                    # time.sleep(.5)

        start_dropdown = widgets.Dropdown(
            description="Start city: ",
            options=sorted(list(node_colors.keys())),
            value="Arad",
        )
        display(start_dropdown)

        end_dropdown = widgets.Dropdown(
            description="Goal city: ",
            options=sorted(list(node_colors.keys())),
            value="Fagaras",
        )
        display(end_dropdown)

        button = widgets.ToggleButton(value=False)
        button_visual = widgets.interactive(visualize_callback, visualize=button)
        display(button_visual)

        slider = widgets.IntSlider(min=0, max=1, step=1, value=0)
        slider_visual = widgets.interactive(slider_callback, iteration=slider)
        display(slider_visual)


# Visualization functions for specific problem types


def plot_NQueens(solution):
    """Visualize N-Queens problem solution on a chessboard.

    Displays queen positions on a checkered board with actual queen images.
    Works with both CSP solutions (dictionary format) and search solutions
    (list format) for the N-Queens problem.

    Args:
        solution (dict or list): Queen positions
            - dict format: {col: row} mapping (from CSP solver)
            - list format: [row_positions] (from search solver)

    Note:
        Assumes 8x8 board and requires 'images/queen_s.png' file.
        Automatically handles different solution formats from various solvers.
        Creates checkered pattern with queens overlaid at correct positions.
    """
    n = len(solution)
    # Create alternating checkered pattern (0=black, 2=white squares)
    board = np.array(
        [2 * int((i + j) % 2) for j in range(n) for i in range(n)]
    ).reshape((n, n))

    # Load queen image for display
    im = Image.open("images/queen_s.png")
    im = np.array(im).astype(float) / 255  # Normalize pixel values

    # Create figure and display checkered board
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title("{} Queens".format(n))
    plt.imshow(board, cmap="binary", interpolation="nearest")

    # Place queen images based on solution format
    if isinstance(solution, dict):
        # CSP solution format: {column: row}
        for k, v in solution.items():
            newax = fig.add_axes(
                [0.064 + (k * 0.112), 0.062 + ((7 - v) * 0.112), 0.1, 0.1], zorder=1
            )
            newax.imshow(im)
            newax.axis("off")
    elif isinstance(solution, list):
        # Search solution format: [row_positions_by_column]
        for k, v in enumerate(solution):
            newax = fig.add_axes(
                [0.064 + (k * 0.112), 0.062 + ((7 - v) * 0.112), 0.1, 0.1], zorder=1
            )
            newax.imshow(im)
            newax.axis("off")
    fig.tight_layout()
    plt.show()


def heatmap(grid, cmap="binary", interpolation="nearest"):
    """Display a 2D grid as a heatmap visualization.

    General-purpose function for visualizing 2D numerical data as color-coded
    heatmaps. Useful for displaying grids, matrices, or spatial data.

    Args:
        grid (array-like): 2D array of numerical values to visualize
        cmap (str): Matplotlib colormap name (default: 'binary')
        interpolation (str): Interpolation method (default: 'nearest')

    Note:
        Creates a square 7x7 inch figure for consistent appearance.
        Values are color-coded according to the specified colormap.
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title("Heatmap")
    plt.imshow(grid, cmap=cmap, interpolation=interpolation)
    fig.tight_layout()
    plt.show()


def gaussian_kernel(length=5, sigma=1.0):
    """Generate a 2D Gaussian kernel for image processing.

    Creates a symmetric Gaussian kernel commonly used for blurring,
    smoothing, or edge detection in image processing applications.

    Args:
        length (int): Kernel size (length x length matrix, default: 5)
        sigma (float): Standard deviation controlling spread (default: 1.0)

    Returns:
        np.ndarray: 2D Gaussian kernel normalized by its maximum value

    Note:
        Larger sigma values create more spread/blur effect.
        Kernel is centered and symmetric around the middle point.
    """
    # Create coordinate arrays from -half_size to +half_size
    ax = np.arange(-length // 2 + 1.0, length // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    # Calculate Gaussian function: exp(-(x^2 + y^2) / (2*sigma^2))
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel


def plot_pomdp_utility(utility):
    """Visualize utility functions for Partially Observable Markov Decision Processes.

    Creates a plot showing utility functions for different actions in a POMDP,
    along with decision boundaries. Helps visualize when each action is optimal
    based on belief states.

    Args:
        utility (dict): Utility data with action keys and value lists
            Expected structure: {'0': [values], '1': [values], '2': [values]}
            where values represent utilities across belief space

    Note:
        Assumes 3 actions: save (green), delete (blue), ask (black).
        Vertical dashed lines show decision boundaries between actions.
        Y-axis shows utility values, X-axis shows belief probability.
    """
    # Extract utility data for each action type
    save = utility["0"][0]
    delete = utility["1"][0]
    ask_save = utility["2"][0]
    ask_delete = utility["2"][-1]

    # Calculate decision boundaries (where utilities intersect)
    left = (save[0] - ask_save[0]) / (save[0] - ask_save[0] + ask_save[1] - save[1])
    right = (delete[0] - ask_delete[0]) / (
        delete[0] - ask_delete[0] + ask_delete[1] - delete[1]
    )

    # Plot utility lines for each action type
    colors = ["g", "b", "k"]  # Green, Blue, Black
    for action in utility:
        for value in utility[action]:
            plt.plot(value, color=colors[int(action)])

    # Add vertical lines showing decision boundaries
    plt.vlines([left, right], -20, 10, linestyles="dashed", colors="c")
    plt.ylim(-20, 13)
    plt.xlim(0, 1)

    # Add action labels showing optimal regions
    plt.text(left / 2 - 0.05, 10, "Save")
    plt.text((right + left) / 2 - 0.02, 10, "Ask")
    plt.text((right + 1) / 2 - 0.07, 10, "Delete")
    plt.show()
