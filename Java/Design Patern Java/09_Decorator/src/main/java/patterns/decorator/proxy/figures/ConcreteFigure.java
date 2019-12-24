package patterns.decorator.proxy.figures;

public class ConcreteFigure implements Figure {

	@Override
	public void draw() {
		System.out.println("drawing concrete figure");
	}

	@Override
	public void move(int dx, int dy) {
		System.out.println("moving concrete figure ("+dx+", "+dy+")");
	}
	
}
