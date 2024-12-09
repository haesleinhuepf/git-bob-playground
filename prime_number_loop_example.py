def is_prime(n):
    """
    Check if a number is prime.

    Parameters
    ----------
    n : int
        The number to check for primality.

    Returns
    -------
    bool
        True if the number is prime, False otherwise.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def print_prime_numbers():
    """
    Print prime numbers between 0 and 100.
    """
    for number in range(101):
        if is_prime(number):
            print(number)

if __name__ == "__main__":
    print_prime_numbers()
