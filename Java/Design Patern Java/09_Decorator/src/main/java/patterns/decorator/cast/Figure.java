package patterns.decorator.cast;

public interface Figure {
	public void move(int dx, int dy);
	public void draw();

	boolean isInstanceOf(Class<?> type);
	<T> T getInstanceOf(Class<T> type);
}
