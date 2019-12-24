package patterns.decorator.proxy.figures;

public class FixationDecorator {

	static Figure create(final Figure f) {
		return Decorators.createDecorator((proxy, m, args) -> {
				if ("move".equals(m.getName())) {
					System.out.println("fixation decorator prohibits moving");
					return null;
				}
				return m.invoke(f, args);
			}
		);
	}

}
