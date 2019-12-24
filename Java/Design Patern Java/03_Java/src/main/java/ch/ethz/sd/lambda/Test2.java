package ch.ethz.sd.lambda;

public class Test2 {

	public static void main(String[] args) {
		Fun<String, Integer> f = s -> s.length();
		System.out.println("f.apply(\"Java\") = " + f.apply("Java"));
	}

}
