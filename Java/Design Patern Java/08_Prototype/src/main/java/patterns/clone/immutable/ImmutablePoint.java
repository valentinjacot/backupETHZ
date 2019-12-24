package patterns.clone.immutable;

public final class ImmutablePoint {
	private final int x, y;

	public ImmutablePoint(int x, int y) {
		this.x = x;
		this.y = y;
	}

	public int getX() {
		return x;
	}

	public int getY() {
		return y;
	}

	public ImmutablePoint withX(int x) {
		return this.x == x ? this : new ImmutablePoint(x, y);
	}

	public ImmutablePoint withY(int y) {
		return this.y == y ? this : new ImmutablePoint(x, y);
	}

	@Override
	public String toString() {
		return String.format("[Point: x=%s, y=%s]", x, y);
	}

	@Override
	public boolean equals(Object p) {
		return p instanceof ImmutablePoint && ((ImmutablePoint)p).x == x && ((ImmutablePoint)p).y == y;
	}
}
