package patterns.command;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class ObjectMethodReference {

	public static void main(String[] args) {
		final List<Integer> list = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
		final MyComparator comparator = new MyComparator();
		
		// Collections.sort(list, comparator); // does not compile 

		// Method reference
		Collections.sort(list, comparator::compare);
		list.forEach(System.out::println);

		// Lambda expression
		Collections.sort(list, (a, b) -> comparator.compare(a, b));
		list.forEach(e -> System.out.println(e));
		
		String s = "Hello";
		Function f = s::contains;
		System.out.println(f.apply("ll"));
	}

	private static class MyComparator {
		public int compare(Integer a, Integer b) {
			return a.compareTo(b);
		}
	}
	
	@FunctionalInterface
	interface Function {
		boolean apply(String s);
	}

}