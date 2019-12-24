package patterns.singleton.registry;

public class Test1 {

	public static void main(String[] args) {
		Registry r = Registry1.getInstance();
		r.register("one", 1);
		r.register("two", 2);
	}

}
