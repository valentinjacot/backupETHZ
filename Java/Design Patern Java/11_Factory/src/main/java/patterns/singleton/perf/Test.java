package patterns.singleton.perf;

public class Test {

	@SuppressWarnings("unused")
	public static void main(String[] args) {
		int N = 1_000_000_000;

		// warm up
		for(int i = 0; i < N; i++) { Singleton1.getInstance(); Singleton3.getInstance(); }

		long start = System.currentTimeMillis();
		for(int i = 0; i < N; i++) {
			Singleton1 s1 = Singleton1.getInstance();
		}
		System.out.println("eager:       " + (System.currentTimeMillis() - start));

		start = System.currentTimeMillis();
		for(int i = 0; i < N; i++) {
			Singleton2 s2 = Singleton2.getInstance();
		}
		System.out.println("lazy sync:   " + (System.currentTimeMillis() - start));

		start = System.currentTimeMillis();
		for(int i = 0; i < N; i++) {
			Singleton3 s3 = Singleton3.getInstance();
		}
		System.out.println("lazy holder: " + (System.currentTimeMillis() - start));

		start = System.currentTimeMillis();
		for(int i = 0; i < N; i++) {
			Singleton4 s4 = Singleton4.INSTANCE;
		}
		System.out.println("enum:        " + (System.currentTimeMillis() - start));

		start = System.currentTimeMillis();
		for(int i = 0; i < N; i++) {
			Singleton5 s4 = Singleton5.getInstance();
		}
		System.out.println("DCL:         " + (System.currentTimeMillis() - start));
	}

}
