package patterns.decorator.proxy;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

public class LoggingHandler implements InvocationHandler {
	private Object target;

	public LoggingHandler(Object t) {
		this.target = t;
	}

	@Override
	public Object invoke(Object proxy, Method m, Object[] args) throws Throwable {
		System.out.print(">> " + m.getName());
		System.out.print("(");
		if (args != null) {
			for (int i = 0; i < args.length; i++) {
				System.out.print(args[i]);
				if (i + 1 < args.length)
					System.out.print(", ");
			}
		}
		System.out.println(")");
		return m.invoke(target, args);
	}

}
