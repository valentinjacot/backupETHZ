package patterns.clone.immutable;

import java.awt.Point;

public final class ImmutableLine {
	private final Point start, end;

	public ImmutableLine(Point start, Point end) {
		this.start = (Point) start.clone();
		this.end = (Point) end.clone();
	}

	public Point getStartPoint() {
		return (Point) start.clone();
	}

	public Point getEndPoint() {
		return (Point) end.clone();
	}

	public ImmutableLine withStartPoint(Point start) {
		return this.start.equals(start) ? this : new ImmutableLine(start, end);
	}

	public ImmutableLine withEndPoint(Point end) {
		return this.end.equals(end) ? this : new ImmutableLine(start, end);
	}

	@Override
	public String toString() {
		return String.format("[Line: start=%s, end=%s]", start, end);
	}
	
//	@Override
//	public Object clone() {	// violates the condition that x.clone() != x 
//			return this;	// which is specified in the specification of
//	}						// Object clone (as a SHOULD requirement).
//	// As an alternative method clone() could also actually create a copy, 
//	// or the method could be omitted altogether (as in class java.lang.String).
}
