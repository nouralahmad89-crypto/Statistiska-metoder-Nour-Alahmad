import numpy as np
import matplotlib.pyplot as plt
import random

rng = np.random.default_rng()

DISTRIBUTIONS = {
    "uniform": lambda: rng.uniform(
        low=random.uniform(-2, 2),
        high=random.uniform(3, 6),
        size=200
    ),

    "normal": lambda: rng.normal(
        loc=random.uniform(-1, 1),
        scale=random.uniform(0.5, 2),
        size=200
    ),

    "binomial": lambda: rng.binomial(
        n=random.randint(5, 20),
        p=random.uniform(0.2, 0.8),
        size=200
    ),

    "negative binomial": lambda: rng.negative_binomial(
        n=random.randint(5, 20),
        p=random.uniform(0.2, 0.8),
        size=200
    ),

    "gamma": lambda: rng.gamma(
        shape=random.uniform(0.5, 3),
        scale=random.uniform(0.5, 2),
        size=200
    ),

    "geometric": lambda: rng.geometric(
        p=random.uniform(0.2, 0.6),
        size=200
    )
}


def plot_histogram(data):
    plt.figure()
    plt.hist(data, bins=20)
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def plot_ogive(data):
    plt.figure()
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, marker="o", linestyle="")
    plt.title("Ogive (Empirical CDF)")
    plt.xlabel("Value")
    plt.ylabel("Cumulative Proportion")
    plt.show()


def quiz_round():
    dist_name = random.choice(list(DISTRIBUTIONS.keys()))
    data = DISTRIBUTIONS[dist_name]()

    plot_type = random.choice(["histogram", "ogive"])

    if plot_type == "histogram":
        plot_histogram(data)
    else:
        plot_ogive(data)

    print("\nWhich distribution is this?")
    print("Options:")
    for name in DISTRIBUTIONS.keys():
        print(" -", name)

    guess = input("\nYour answer: ").strip().lower()

    if guess == dist_name:
        print("✅ Correct!")
        return 1
    else:
        print(f"❌ Incorrect. Correct answer: {dist_name}")
        return 0


def main():
    print("\n📊 Distribution Identification Quiz 📊")
    print("Type Ctrl+C to quit.\n")

    score = 0
    rounds = 0

    try:
        while True:
            score += quiz_round()
            rounds += 1
            print(f"\nScore: {score} / {rounds}")
            input("\nPress Enter for next question...")
    except KeyboardInterrupt:
        print("\n\nQuiz ended.")
        print(f"Final score: {score} / {rounds}")


if __name__ == "__main__":
    main()

