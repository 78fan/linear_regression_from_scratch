import matplotlib.pyplot as plt
import random
from typing import List, Tuple

test_squares = [i + random.randrange(-5000, 5000) for i in range(1000, 11000, 500)]

test_prices = [i + random.randrange(-50000, 50000) for i in range(200000, 500000, 15000)]

def plot_linear_regression(squares: List[int], prices: List[int], line: List[float] = None, line2: List[float] = None):
    plt.figure(figsize=(10, 6))
    plt.scatter(squares, prices, color='blue')
    if line is not None:
        x_min, x_max = min(squares), max(squares)
        y_min = line[0] * x_min + line[1]
        y_max = line[0] * x_max + line[1]
        plt.plot([x_min, x_max], [y_min, y_max], color='red', linewidth=2)
    if line2 is not None:
        x_min, x_max = min(squares), max(squares)
        y_min = line2[0] * x_min + line2[1]
        y_max = line2[0] * x_max + line2[1]
        plt.plot([x_min, x_max], [y_min, y_max], color='black', linewidth=2)
    plt.title('House Prices vs Square Footage', fontsize=16)
    plt.xlabel('Square Footage', fontsize=14)
    plt.ylabel('Price ($)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def square_descent(line: List[float], squares: List[int], prices: List[int], step: float):
    point = random.randrange(len(squares))
    square = squares[point]
    price = prices[point]
    predicted_price = line[0] * square + line[1]
    line[0] += step * square * (price - predicted_price)
    line[1] += step * (price - predicted_price)

def absolute_descent(line: List[float], squares: List[int], prices: List[int], step: float):
    point = random.randrange(len(squares))
    square = squares[point]
    price = prices[point]
    predicted_price = line[0] * square + line[1]
    if price > predicted_price:
        line[0] += step * square
        line[1] += step
    else:
        line[0] -= step * square
        line[1] -= step


def linear_regression(squares: List[int], prices: List[int], step: float, steps: int, mode:str) -> List[float]:
    assert len(squares) == len(prices)
    line = [random.uniform(-1, 1), random.uniform(-1, 1)]
    for _ in range(steps):
        if mode == "SQUARE":
            square_descent(line, squares, prices, step)
        elif mode == "ABSOLUTE":
            absolute_descent(line, squares, prices, step)
        else:
            raise ValueError("Incorrect descent mode")
    return line


def normalise_data(squares: List[int], prices: List[int]) -> Tuple[int, int, int, int]:
    max_square = max(squares, key=lambda x: abs(x))
    max_price = max(prices, key=lambda x: abs(x))
    squares = [square/max_square for square in squares]
    prices = [price/max_price for price in prices]
    return squares, prices, max_square, max_price


if __name__ == '__main__':
    norm_squares, norm_prices, scale1, scale2 = normalise_data(test_squares, test_prices)
    regression_line1 = linear_regression(norm_squares, norm_prices, 0.01, 1_000_000, "SQUARE")
    regression_line1[0] *= scale2/scale1
    regression_line1[1] *= scale2
    regression_line2 = linear_regression(norm_squares, norm_prices, 0.01, 1_000_000, "ABSOLUTE")
    regression_line2[0] *= scale2 / scale1
    regression_line2[1] *= scale2
    plot_linear_regression(test_squares, test_prices, regression_line1, regression_line2)