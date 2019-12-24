package patterns.decorator.delegation;

public abstract class Figure {
	
	public abstract void moveImpl(int dx, int dy);

	public abstract void drawImpl();
	
	public final void draw() {
		if (parent != null) {
			parent.draw();
		} else {
			drawImpl();
		}
	}

	public final void move(int dx, int dy) {
		if (parent != null) {
			parent.move(dx, dy);
		} else {
			moveImpl(dx, dy);
		}
	}

	private Figure parent;
	public final Figure getParent() { return parent; }
	public void setParent(Figure parent) {
		this.parent = parent;
	}
	
}
