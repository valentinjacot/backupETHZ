package patterns.clone.immutable;

public final class ImmutableLine2 {
	private final ImmutablePoint start, end;

	public ImmutableLine2(ImmutablePoint start, ImmutablePoint end) {
		this.start = start;
		this.end = end;
	}

	public ImmutablePoint getStartPoint() {
		return start;
	}

	public ImmutableLine2 withStartPoint(ImmutablePoint start) {
		return this.start.equals(start) ? this : new ImmutableLine2(start, end);
	}

	public ImmutablePoint getEndPoint() {
		return end;
	}

	public ImmutableLine2 withEndPoint(ImmutablePoint end) {
		return this.end.equals(end) ? this : new ImmutableLine2(start, end);
	}

	@Override
	public String toString() {
	   return String.format("[Line: start=%s, end=%s]", start, end);
	}
}
