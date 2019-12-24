package patterns.command;

import java.util.List;

public class StaticMethodReference {

	public static void main(String[] args) {
		List<Integer> list = List.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

		// Method reference
		list.forEach(StaticMethodReference::print);

		// Lambda expression
		list.forEach(number -> StaticMethodReference.print(number));

		// normal
		for (int number : list) {
			StaticMethodReference.print(number);
		}

		Function f = String::format;
		System.out.println(f.apply("%s/%s", 1, 2));
	}

	public static void print(final int number) {
		System.out.println("I am printing: " + number);
	}

	@FunctionalInterface
	interface Function {
		String apply(String s, Object e1, Object e2);
	}
}