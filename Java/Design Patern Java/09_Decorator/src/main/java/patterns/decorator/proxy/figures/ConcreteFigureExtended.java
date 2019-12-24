package patterns.decorator.proxy.figures;

public class ConcreteFigureExtended extends ConcreteFigure {

	@Override
	public void draw() {
		System.out.println("drawing extended concrete figure");
	}
	
	public boolean equals(Object x) {
		System.out.println("equals on " + this + " with " + x);
		return this == x;
	}

}
