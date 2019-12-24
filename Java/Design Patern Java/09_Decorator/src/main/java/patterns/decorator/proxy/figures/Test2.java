package patterns.decorator.proxy.figures;

import java.util.LinkedList;
import java.util.List;

public class Test2 {
	
	public static void main(String[] args) {
		Figure f = new ConcreteFigureExtended();
		System.out.println(f);
		System.out.println(System.identityHashCode(f));

		f = BorderDecorator.create(f);
		f = FixationDecorator.create(f);
		System.out.println(f);
		System.out.println(System.identityHashCode(f));
		
		System.out.println(f.equals(f));
		
		List<Figure> list = new LinkedList<>();
		list.add(f);
		System.out.println(list.size());
		list.remove(f);
		System.out.println(list.size());
	}

}
