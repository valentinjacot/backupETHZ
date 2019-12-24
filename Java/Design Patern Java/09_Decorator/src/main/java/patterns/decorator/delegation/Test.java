package patterns.decorator.delegation;

public class Test {

	public static void main(String[] args) {
		Figure f = new ConcreteFigure();
		
		f = new AnimationDecorator(f);
//		f = new FixationDecorator(f);
		f = new BorderDecorator(f);
	}

}
