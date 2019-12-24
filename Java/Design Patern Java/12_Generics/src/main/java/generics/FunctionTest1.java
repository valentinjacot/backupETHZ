package generics;

import java.util.function.Function;

@SuppressWarnings("unused")
public class FunctionTest1 {

	static class A {}
	static class B extends A {}
	static class C extends B {}

	public static void main(String[] args) {
		
		// function represents a function that accepts one argument of type B and returns a result of type B:
		Function<B, B> function;
		
//		function = FunctionTest::applyAA;	// return type incompatible
		function = FunctionTest1::applyAB;
		function = FunctionTest1::applyAC;
		
//		function = FunctionTest::applyBA;	// return type incompatible
		function = FunctionTest1::applyBB;
		function = FunctionTest1::applyBC;

//		function = FunctionTest::applyCA;	// not applicable (return type and argument incompatible)
//		function = FunctionTest::applyCB;	// not applicable (argument incompatible)
//		function = FunctionTest::applyCC;	// not applicable (argument incompatible)
		
	}
	
	static A applyAA(A a) { return a; }
	static B applyAB(A a) { return new B(); }
	static C applyAC(A a) { return new C(); }
	static A applyBA(B b) { return b; }
	static B applyBB(B b) { return b; }
	static C applyBC(B b) { return new C(); }
	static A applyCA(C c) { return c; }
	static B applyCB(C c) { return c; }
	static C applyCC(C c) { return c; }
	
}
