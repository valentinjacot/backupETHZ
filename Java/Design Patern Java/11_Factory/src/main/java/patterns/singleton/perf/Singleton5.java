package patterns.singleton.perf;

public class Singleton5 {

	private volatile static Singleton5 instance;

	public static Singleton5 getInstance() {
		if (instance == null) {
			synchronized (Singleton5.class) {
				if (instance == null) {
					instance = new Singleton5();
				}
			}
		}
		return instance;
	}

	private Singleton5() { /* initialization */
	}
	
	// other methods
}
