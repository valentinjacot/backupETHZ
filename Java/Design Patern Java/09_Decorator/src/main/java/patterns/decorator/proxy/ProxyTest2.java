package patterns.decorator.proxy;

import java.lang.reflect.Proxy;
import java.util.LinkedList;
import java.util.List;

public class ProxyTest2 {

	public static void main(String[] args) {
		// real object
		String s1 = "Hello";
		
		// String proxy
		CharSequence s2 = (CharSequence)Proxy.newProxyInstance(
				String.class.getClassLoader(),
				new Class[] { CharSequence.class },
				new LoggingHandler(s1)
		);

		System.out.println("s1 instanceof CharSequence" + (s1 instanceof CharSequence));
		System.out.println("s2 instanceof CharSequence" + (s2 instanceof CharSequence));

		List<CharSequence> list = new LinkedList<>();
		list.add(s2);
		System.out.println(list.contains(s2));
		System.out.println(list.contains(s1));

		list.clear();
		list.add(s1);
		System.out.println(list.contains(s2));
		System.out.println(list.contains(s1));
	}

}
