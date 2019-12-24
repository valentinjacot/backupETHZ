package patterns.command.gof;

public class Model {
	public void addFigure(Figure f) {
		System.out.printf("%s.addFigure(%s)%n", this, f);
	}
}
