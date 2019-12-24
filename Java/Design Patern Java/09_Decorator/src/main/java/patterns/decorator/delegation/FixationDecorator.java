package patterns.decorator.delegation;

public class FixationDecorator extends AbstractDecorator {

	public FixationDecorator(Figure inner) {
		super(inner);
	}

	@Override
	public void moveImpl(int dx, int dy) {
		System.out.println("fixation decorator prohibits moving");
	}

}
