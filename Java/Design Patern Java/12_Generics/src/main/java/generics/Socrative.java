package generics;

import java.util.ArrayList;

@SuppressWarnings("unused")
public class Socrative {
	static class A {}
	static class B extends A {}
	
	public static void main(String[] args) {
		// 1)
		A[] array = new B[1];
		
		// 2)
//		ArrayList<A> list = new ArrayList<B>();

		ArrayList<? extends A> lista = new ArrayList<B>();
		A a = lista.get(0);
		// lista.add(new A());
		
		ArrayList<? super B> listb = new ArrayList<A>();
		// B b = listb.get(0);
		listb.add(new B());
	}
}
