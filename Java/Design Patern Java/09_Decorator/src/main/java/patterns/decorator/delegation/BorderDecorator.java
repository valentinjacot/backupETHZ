package patterns.decorator.delegation;

public class BorderDecorator extends AbstractDecorator {

	public BorderDecorator(Figure inner) {
		super(inner);
	}

	@Override
	public void drawImpl() {
		System.out.println("draw border");
		super.drawImpl();
	}

}
