"""Provide the primary functions."""


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


def mean(num_list):
    '''
    Calculate the average of a list of numbers

    Parameters
    _____
    num_list: list
        The list to take average of

    Returns
    ____
    mean_list: float
        The mean of the list

    Examples
    ____
    >> mean([1,2,3])
    2.0
    '''
    mean_list = sum(num_list) / len(num_list)
    return mean_list


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
