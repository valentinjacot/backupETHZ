package patterns.decorator.cast;

public class Test {
	
	public static void main(String[] args) {
		Figure f = new ConcreteFigureExtended();
		f = new BorderDecorator(f);
		f = new AnimationDecorator(f);
		System.out.println(f);
		
		System.out.println("\nisInstanceOf(ConcreteFigure.class)");
		System.out.println(f.isInstanceOf(ConcreteFigure.class));
		System.out.println(f.getInstanceOf(ConcreteFigure.class));
		
		System.out.println("\nisInstanceOf(ConcreteFigureExtended.class)");
		System.out.println(f.isInstanceOf(ConcreteFigureExtended.class));
		System.out.println(f.getInstanceOf(ConcreteFigureExtended.class));
		
		System.out.println("\nisInstanceOf(BorderDecorator.class)");
		System.out.println(f.isInstanceOf(BorderDecorator.class));
		System.out.println(f.getInstanceOf(BorderDecorator.class));
		
		System.out.println("\nisInstanceOf(AnimationDecorator.class)");
		System.out.println(f.isInstanceOf(AnimationDecorator.class));
		System.out.println(f.getInstanceOf(AnimationDecorator.class));

		System.out.println("\nisInstanceOf(FixationDecorator.class)");
		System.out.println(f.isInstanceOf(FixationDecorator.class));
		try { System.out.println(f.getInstanceOf(FixationDecorator.class)); }
		catch(ClassCastException e) { System.out.println(e); }

	}

}
