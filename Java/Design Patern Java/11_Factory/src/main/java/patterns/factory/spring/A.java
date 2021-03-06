package patterns.factory.spring;

public class A {
	private B b;
	private String value;
	
	public A() {
		System.out.println("A() called");
	}


	public B getB() {
		return b;
	}

	public void setB(B b) {
		System.out.println("A.setB called");
		this.b = b;
	}
	

	public String getValue() {
		return value;
	}

	public void setValue(String value) {
		this.value = value;
	}
}
