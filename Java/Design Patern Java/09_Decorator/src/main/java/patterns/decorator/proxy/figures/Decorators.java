package patterns.decorator.proxy.figures;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;

public class Decorators {
	
	static Figure createDecorator(InvocationHandler handler) {
		return (Figure)Proxy.newProxyInstance(
			Figure.class.getClassLoader(),
			new Class[] { Figure.class },
			handler
		);
	}

}
