package generics;

import java.util.function.Function;

@SuppressWarnings("unused")
public class FunctionTest2 {

	static class A {}
	static class B extends A {}
	static class C extends B {}
	
	public static void main(String[] args) {

		// function represents a function that accepts one argument of type B and returns a result of type B:
		Function<? super B, ? extends B> function;
		// Function<B, B> function;
		
//		function = (A a) -> new A();	// return type incompatible
		function = (A a) -> new B();
		function = (A a) -> new C();
		
//		function = (B a) -> new A();	// return type incompatible
		function = (B a) -> new B();
		function = (B a) -> new C();
		
//		function = (C a) -> new A();	// not applicable (return type and argument incompatible)
//		function = (C a) -> new B();	// not applicable (argument incompatible)
//		function = (C a) -> new C();	// not applicable (argument incompatible)
	}
	
}
