package patterns.decorator.proxy;

import java.lang.reflect.Proxy;
import java.util.Collection;
import java.util.HashSet;

public class ProxyTest {

	@SuppressWarnings("unchecked")
	public static void main(String[] args) {
		// real object
		String s = "Hello";
		
		// String proxy
		CharSequence obj = (CharSequence)Proxy.newProxyInstance(
				String.class.getClassLoader(),
				new Class[] { CharSequence.class },
				new LoggingHandler(s)
		);
		System.out.println("Logging of a CharSequence object");
		obj.length();
		obj.charAt(0);
		
		System.out.println();

		Collection<String> c = (Collection<String>)Proxy.newProxyInstance(
				Collection.class.getClassLoader(),
				new Class[] { Collection.class },
				new LoggingHandler(new HashSet<String>())
			);
		
		System.out.println("Logging of a hash set");
		c.size();
		c.add("Test");
	}

}
