package patterns.singleton.registry;

public class Test2 {

	public static void main(String[] args) {
		Registry r = Registry2.INSTANCE;
		r.register("one", 1);
		r.register("two", 2);
	}

}
