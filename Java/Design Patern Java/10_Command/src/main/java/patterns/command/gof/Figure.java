package patterns.command.gof;

public class Figure {
	public void move(int dx, int dy) {
		System.out.printf("%s.move(dx=%d, dy=%d)%n", this, dx, dy);
	}
}
