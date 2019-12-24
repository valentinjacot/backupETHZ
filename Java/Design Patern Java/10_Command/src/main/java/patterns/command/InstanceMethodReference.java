package patterns.command;

import java.util.Arrays;
import java.util.List;

public class InstanceMethodReference {

	public static void main(String[] args) {
		final List<Person> people = Arrays.asList(new Person("Romeo"), new Person("Julia"));

		// Method reference
		people.forEach(Person::printName);

		// Lambda expression
		people.forEach(person -> person.printName());

		// normal
		for (Person person : people) {
			person.printName();
		}

		String s = "Hello";
		Function f = String::contains;
		System.out.println(f.apply(s, "ll"));
	}

	private static class Person {
		private final String name;

		public Person(String name) {
			this.name = name;
		}

		public void printName() {
			System.out.println(name);
		}
	}

	@FunctionalInterface
	interface Function {
		boolean apply(String s, String pat);
	}

}