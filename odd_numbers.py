#!/usr/bin/env python3

def odd_numbers(start=0, end=20):
    """
    Compute odd numbers within a closed interval.

    Parameters
    ----------
    start : int, optional
        Start of the interval (inclusive). Default is 0.
    end : int, optional
        End of the interval (inclusive). Default is 20.

    Returns
    -------
    list of int
        List of odd integers in [start, end].
    """
    return [n for n in range(start, end + 1) if n % 2 == 1]


if __name__ == "__main__":
    for n in odd_numbers(0, 20):
        print(n)
