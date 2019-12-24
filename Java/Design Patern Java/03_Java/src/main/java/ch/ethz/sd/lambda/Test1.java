package ch.ethz.sd.lambda;

import java.util.function.Function;

public class Test1 {
	
	public static Function<Integer, Integer> addTo(int y) {
		// y = 4;
		return x -> x + y;
	}
	
	public static void main(String[] args) {
		Function<Integer, Integer> f = addTo(1);
		System.out.println(f.apply(2));
	}
	
}
