package patterns.singleton.perf;

public final class Singleton2 {
	
	private Singleton2() { }

	private static Singleton2 instance = null;

	public static synchronized Singleton2 getInstance() {
		if (instance == null)
			instance = new Singleton2();
		return instance;
	}
}
