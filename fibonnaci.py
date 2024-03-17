def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a, end=' ')
        a, b = b, a+b
    print()  # for new line


if __name__ == "__main__":
    N = int(input("Enter the length of the Fibonacci sequence: "))
    fibonacci(N)
