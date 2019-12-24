package patterns.command;

import java.util.function.Function;

public class FunctionTest {
	
	interface Function2<B> {
		void apply(B arg);
	}

	@SuppressWarnings("unused")
	public static void main(String[] args) {
//		Function<? super B, ? extends B> function;
		Function<B, B> function;
		
		function = (B b) -> b;
		function = (B b) -> new C();
//		function = (A a) -> new C();	// => Lambda expression's parameter a is expected to be of type FunctionTest.B

//		function = FunctionTest::applyAA;
		function = FunctionTest::applyAB;
		function = FunctionTest::applyAC;
		
//		function = FunctionTest::applyBA;
		function = FunctionTest::applyBB;
		function = FunctionTest::applyBC;

//		function = FunctionTest::applyCA;
//		function = FunctionTest::applyCB;
//		function = FunctionTest::applyCC;
		
		Function2<B> function2;
//		function2 = (B b) -> b;
		function2 = (B b) -> Function.identity().apply(b);
		function2 = (B b) -> new A();
		function2 = (B b) -> new B();
		function2 = (B b) -> new C();
		//function2 = (A a) -> new C();	// => Lambda expression's parameter a is expected to be of type FunctionTest.B
		//function2 = (C c) -> new B();	// => Lambda expression's parameter c is expected to be of type FunctionTest.B

		function2 = FunctionTest::applyAA;
		function2 = FunctionTest::applyAB;
		function2 = FunctionTest::applyAC;
		
		function2 = FunctionTest::applyBA;
		function2 = FunctionTest::applyBB;
		function2 = FunctionTest::applyBC;

//		function2 = FunctionTest::applyCA;
//		function2 = FunctionTest::applyCB;
//		function2 = FunctionTest::applyCC;

	}
	
	static A applyAA(A a) { return a;}
	static B applyAB(A a) { return new B(); }
	static C applyAC(A a) { return new C();}
	static A applyBA(B b) { return b;}
	static B applyBB(B b) { return b; }
	static C applyBC(B b) { return new C();}
	static A applyCA(C c) { return c;}
	static B applyCB(C c) { return c; }
	static C applyCC(C c) { return c;}
	
	static class A {}
	static class B extends A {}
	static class C extends B {}
}
