package patterns.decorator.delegation;

public class ConcreteFigure extends Figure {
	private int x, y;

	@Override
	public void drawImpl() {
		System.out.printf("ConcreteFigure::draw: drawing concrete figure at (%d, %d)\n", x, y);
	}

	@Override
	public void moveImpl(int dx, int dy) {
		x += dx;
		y += dy;
		System.out.printf("ConcreteFigure::move: moving concrete figure (%d, %d)\n", dx, dy);
	}

}
