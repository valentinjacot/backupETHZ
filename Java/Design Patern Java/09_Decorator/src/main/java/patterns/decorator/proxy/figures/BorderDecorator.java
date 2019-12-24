package patterns.decorator.proxy.figures;

public class BorderDecorator {

	static Figure create(final Figure f) {
		return Decorators.createDecorator((proxy, m, args) -> {
			if (m.getName().equals("draw")) {
				System.out.println("draw border");
			}
			return m.invoke(f, args);
		});
	}

}
