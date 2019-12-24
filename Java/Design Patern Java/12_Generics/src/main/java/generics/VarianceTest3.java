package generics;

@SuppressWarnings("unused")
public class VarianceTest3 {

	static class A {}
	static class B extends A {}
	static class C extends B {}
	
	static class Cage<T extends A> {
		private T x;
		public T getContent() { return x; }
		public void setContent(T a) { x = a; }
		public T replaceContent(T a) { T ret = x; x = a; return ret; }
	}
	
	public static void readContent(Cage<? extends B> cage) {
		B b = cage.getContent();
		// cage.setContent(new B());
		cage.setContent(null);
		// a = cage.replaceContent(new B());
		b = cage.replaceContent(null);
	}
	
	public static void writeContent(Cage<? super B> cage) {
		// B b = cage.getContent();
		cage.setContent(new B());
		// b = cage.replaceContent(new B());
	}
	
	public static void readWriteContent(Cage<B> cage) {
		B b = cage.getContent();
		cage.setContent(new B());
		cage.setContent(null);
		b = cage.replaceContent(new B());
		b = cage.replaceContent(null);
	}
	

	public static void main(String[] args) {
		Cage<A> ca = new Cage<>();
		Cage<B> cb = new Cage<>();
		Cage<C> cc = new Cage<>();
		
		// readAnimal(ca);
		readContent(cb);
		readContent(cc);
		
		writeContent(ca);
		writeContent(cb);
		// writeContent(cc);
		
		System.out.println("done");
	}
}
