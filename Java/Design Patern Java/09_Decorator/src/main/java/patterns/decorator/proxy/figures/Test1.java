package patterns.decorator.proxy.figures;

public class Test1 {
	
	public static void main(String[] args) {
		Figure f = new ConcreteFigureExtended();
		System.out.println(f);
		System.out.println(System.identityHashCode(f));

		f = BorderDecorator.create(f);
		f = FixationDecorator.create(f);
		System.out.println(f);
		System.out.println(System.identityHashCode(f));
		
		System.out.println("\nf.move(1,2)");
		f.move(1,2);
		
		System.out.println("\nf.draw()");
		f.draw();
	}

}
