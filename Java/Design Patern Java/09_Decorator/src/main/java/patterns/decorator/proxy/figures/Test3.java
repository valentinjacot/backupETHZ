package patterns.decorator.proxy.figures;

import java.util.HashSet;
import java.util.Set;

public class Test3 {
	
	public static void main(String[] args) {
		Figure f = new ConcreteFigureExtended();
		System.out.println(f);
		System.out.println(System.identityHashCode(f));

		f = BorderDecorator.create(f);
		f = FixationDecorator.create(f);
		System.out.println(f);
		System.out.println(System.identityHashCode(f));
		
		System.out.println(f.equals(f));
		
		Set<Figure> set = new HashSet<>();
		set.add(f);
		System.out.println(set.size());
		set.remove(f);
		System.out.println(set.size());
	}

}
