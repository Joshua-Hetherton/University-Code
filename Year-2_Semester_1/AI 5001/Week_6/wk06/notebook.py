from inspect import getsource


import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from IPython.display import display


from learning import DataSet


# ______________________________________________________________________________
# Magic Words - Utility functions for displaying algorithm pseudocode and source code


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
# Iris Dataset Visualization - 3D scatter plots for the classic Iris dataset


def show_iris(i=0, j=1, k=2):
    """Plots the iris dataset in a 3D scatter plot.

    Creates a 3D visualization of the Iris dataset with customizable axes.
    Each species (setosa, virginica, versicolor) is displayed with different
    colors and markers for easy distinction.

    Args:
        i (int): Index for first feature axis (0-3, default: 0 - Sepal Length)
        j (int): Index for second feature axis (0-3, default: 1 - Sepal Width)
        k (int): Index for third feature axis (0-3, default: 2 - Petal Length)

    Note:
        The four iris features are: Sepal Length, Sepal Width, Petal Length, Petal Width
        Colors: blue squares (setosa), green triangles (virginica), red circles (versicolor)
    """

    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Load Iris dataset and separate by species
    iris = DataSet(name="iris")
    buckets = iris.split_values_by_classes()

    # Define feature names for axis labels
    features = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    f1, f2, f3 = features[i], features[j], features[k]

    # Extract feature values for each species
    # Setosa species data
    a_setosa = [v[i] for v in buckets["setosa"]]
    b_setosa = [v[j] for v in buckets["setosa"]]
    c_setosa = [v[k] for v in buckets["setosa"]]

    # Virginica species data
    a_virginica = [v[i] for v in buckets["virginica"]]
    b_virginica = [v[j] for v in buckets["virginica"]]
    c_virginica = [v[k] for v in buckets["virginica"]]

    # Versicolor species data
    a_versicolor = [v[i] for v in buckets["versicolor"]]
    b_versicolor = [v[j] for v in buckets["versicolor"]]
    c_versicolor = [v[k] for v in buckets["versicolor"]]

    # Plot each species with distinct colors and markers
    # Blue squares for setosa, green triangles for virginica, red circles for versicolor
    for c, m, sl, sw, pl in [
        ("b", "s", a_setosa, b_setosa, c_setosa),
        ("g", "^", a_virginica, b_virginica, c_virginica),
        ("r", "o", a_versicolor, b_versicolor, c_versicolor),
    ]:
        ax.scatter(sl, sw, pl, c=c, marker=m)

    # Set axis labels based on selected features
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_zlabel(f3)

    plt.show()


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


############################################################################################################
#####################        Search Algorithm Visualization Functions         ####################
############################################################################################################


# Visualization functions for specific problem types


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
