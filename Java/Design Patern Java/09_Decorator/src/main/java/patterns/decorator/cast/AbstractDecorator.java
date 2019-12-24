package patterns.decorator.cast;

public class AbstractDecorator implements Figure {
	private final Figure inner;

	public Figure getInner() {
		return inner;
	}

	public AbstractDecorator(Figure inner) {
		this.inner = inner;
	}

	@Override
	public void draw() {
		inner.draw();
	}

	@Override
	public void move(int dx, int dy) {
		inner.move(dx,dy);
	}

	@Override
	public final boolean isInstanceOf(Class<?> type) {
		return type.isAssignableFrom(this.getClass()) || inner.isInstanceOf(type);
	}

	@Override
	public final <T> T getInstanceOf(Class<T> type) {
		if(type.isAssignableFrom(this.getClass())){
			return type.cast(this); // checked version of (T)this
		} else {
			return inner.getInstanceOf(type);
		}
	}

}
