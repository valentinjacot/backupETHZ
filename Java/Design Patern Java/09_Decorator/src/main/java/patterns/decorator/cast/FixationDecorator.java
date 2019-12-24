package patterns.decorator.cast;

public class FixationDecorator extends AbstractDecorator {

	public FixationDecorator(Figure inner) {
		super(inner);
	}

	@Override
	public void move(int dx, int dy) {
		System.out.println("fixation decorator prohibits moving");
	}

}
