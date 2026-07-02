"""Smoke test — verify the Rust + PyO3 build works."""

from fgh import hello, add


def test_hello():
    msg = hello()
    assert "FGH" in msg
    print(f"  hello() → {msg}")


def test_add():
    assert add(2, 3) == 5
    print("  add(2, 3) == 5  ✓")


if __name__ == "__main__":
    test_hello()
    test_add()
    print("\nFGH smoke test passed!")
