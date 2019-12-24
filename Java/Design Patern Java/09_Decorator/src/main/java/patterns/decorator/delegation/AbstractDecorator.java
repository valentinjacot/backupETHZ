package patterns.decorator.delegation;

public class AbstractDecorator extends Figure {
	private final Figure inner;

	public Figure getInner() {
		return inner;
	}

	public AbstractDecorator(Figure inner) {
		this.inner = inner;
		inner.setParent(this);
	}

	@Override
	public void drawImpl() {
		inner.drawImpl();
	}

	@Override
	public void moveImpl(int dx, int dy) {
		inner.moveImpl(dx, dy);
	}

}
