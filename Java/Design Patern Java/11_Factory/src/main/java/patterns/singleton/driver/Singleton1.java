package patterns.singleton.driver;

import java.io.*;

@SuppressWarnings("serial")
public class Singleton1 implements Serializable, Driver {
	private static Singleton1 instance;

	private Singleton1() {
		System.out.println(this + " Singleton() called");
	}

	@Override
	public void playSong(File file) {
		System.out.println("Yesterday, all my troubles seemed so far away....");
	}

	public static synchronized Singleton1 getInstance() {
		if (instance == null) {
			instance = new Singleton1();
		}
		return instance;
	}

	private Object readResolve() {
		// during object input, convert this deserialized singleton into the
		// proper singleton instance defined in the singleton class.
		System.out.println(this + " readResolve");
		return getInstance();
	}

//	private Object writeReplace() {
//		System.out.println(this + " writeReplace");
//		return this;
//	}
}
