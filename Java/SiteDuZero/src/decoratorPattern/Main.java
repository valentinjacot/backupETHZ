package decoratorPattern;

public class Main {

	public static void main(String[] args) {
		Patisserie pat = new CoucheChocolat(new CoucheBiscuit(new CoucheBiscuit(new CoucheCaramel((new Gateau())))));
		System.out.println(pat.preparer());
		Patisserie mil = new CoucheChocolat(new CoucheBiscuit(new Millfeuille()));
		System.out.println(mil.preparer());

	}

}
