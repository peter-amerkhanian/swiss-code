import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib

def show_all_xticks(ax: matplotlib.axes.Axes, labs: pd.Index) -> matplotlib.axes.Axes:
    """
    Sets all x-ticks on a Matplotlib axis object and labels them with the provided list of labels.

    This function ensures that all x-ticks are displayed and labeled as specified, making it easier to read and interpret the x-axis of the plot. The labels are displayed horizontally.

    Parameters:
    ax (matplotlib.axes.Axes): The axis object on which to set the x-ticks and labels.
    labs (list of str): A list of labels to set on the x-axis. The number of labels should correspond to the number of ticks.

    Returns:
    matplotlib.axes.Axes: The modified axis object with updated x-ticks and labels.
    """
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=0)
    return ax

def comma_formatter() -> ticker.FuncFormatter:
    """
    Creates a custom Matplotlib tick formatter that formats axis ticks with commas.

    This formatter converts axis tick values to integers and formats them with commas for thousands 
    separators. This can be useful for improving the readability of plots with large numerical values.

    Returns:
    --------
    ticker.FuncFormatter:
        A Matplotlib FuncFormatter object that applies the comma formatting to axis ticks.
    """
    def comma(x: float, pos) -> str:
        """Format the tick value `x` with commas. The parameter `pos` is not used."""
        return '{:,}'.format(int(x))

    return ticker.FuncFormatter(comma)