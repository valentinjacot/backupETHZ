package patterns.builder;

@SuppressWarnings("unused")
public class PizzaTest {

	public static void main(String[] args) {
		Pizza pizza = Pizza.builder(12)
				.cheese(true)
				.pepperoni(true)
				.bacon(true)
				.build();

	}

}
