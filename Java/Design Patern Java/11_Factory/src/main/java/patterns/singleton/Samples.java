package patterns.singleton;

import java.awt.Toolkit;
import java.util.logging.Logger;

public class Samples {

	public static void main(String[] args) {
		Runtime r1 = Runtime.getRuntime();
		Runtime r2 = Runtime.getRuntime();
		System.out.println("Runtime: " + (r1 == r2));


		Toolkit t1 = Toolkit.getDefaultToolkit();
		Toolkit t2 = Toolkit.getDefaultToolkit();
		System.out.println("Toolkit: " + (t1 == t2));


		Logger logger1 = Logger.getLogger("patterns");
		Logger logger2 = Logger.getLogger("other");
		Logger logger3 = Logger.getLogger("patterns");
		Logger logger4 = Logger.getLogger("other");
		System.out.println("Patterns Logger:  " + (logger1 == logger3));
		System.out.println("Other Logger:  " + (logger2 == logger4));
		System.out.println("Different Loggers:  " + (logger3 == logger4));


		Class<?> c1 = Samples.class;
		Class<?> c2 = new Samples().getClass();
		System.out.println("Class:   " + (c1 == c2));
	}

}
