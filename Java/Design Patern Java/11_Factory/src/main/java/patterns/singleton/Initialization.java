package patterns.singleton;

public class Initialization {
	public static void main(String[] args) throws Exception {
		System.out.println("A.x = " + A.x);
		System.out.println("B.x = " + B.x);
	}
}

class A {
	static { System.out.println("A()"); }
	static int x;
	static {
		x = B.x + 1;
	}
	static { System.out.println("A() exit"); }
}

class B {
	static { System.out.println("B()"); }
	static int x;
	static {
		x = A.x + 1;
	}
	static { System.out.println("B() exit"); }
}





// Class.forName("patterns.singleton.B");
// Class.forName(B.class.getName());











/*
A()
B()
A.x = 2
B.x = 1

B()
A()
A.x = 1
B.x = 2

*/