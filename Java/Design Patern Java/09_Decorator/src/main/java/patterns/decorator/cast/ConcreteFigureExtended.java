package patterns.decorator.cast;

public class ConcreteFigureExtended extends ConcreteFigure {

	@Override
	public void draw() {
		System.out.println("drawing extended concrete figure");
	}

	@Override
	public void move(int dx, int dy) {
		System.out.println("moving extended concrete figure ("+dx+", "+dy+")");
	}

}
