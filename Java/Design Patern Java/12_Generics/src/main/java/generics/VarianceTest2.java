package generics;

@SuppressWarnings("unused")
public class VarianceTest2 {

	static class Animal {}
	static class Bird extends Animal {}
	
	static class Cage<A extends Animal> {
		private A animal;
		public A getAnimal() { return animal; }
		public void storeAnimal(A a) { animal = a; }
		public A replaceAnimal(A a) { A ret = animal; animal = a; return ret; }
	}
	
	public void readAnimal(Cage<? extends Animal> cage) {
		Animal a = cage.getAnimal();
		// cage.storeAnimal(new Animal());
		// a = cage.replaceAnimal(new Animal());

		// special situation for null argument
		cage.storeAnimal(null);
		a = cage.replaceAnimal(null);
	}
	
	public void setAnimal(Cage<? super Bird> cage) {
		// Bird b = cage.getAnimal();
		cage.storeAnimal(new Bird());
		// Bird b = cage.replaceAnimal(new Bird());
	}
	
	
	public static void main(String[] args) {
		Cage<? extends Animal> cage1 = new Cage<Bird>();
		Animal a = cage1.getAnimal();
//		cage1.storeAnimal(new Animal());

		Cage<? super Bird> cage2 = new Cage<Animal>();
//		Bird b = cage2.getAnimal();
		Animal x = cage2.getAnimal();
		cage2.storeAnimal(new Bird());
	}
}




// Der Aurfuf von getAnimal auf cage2 funkioniert, denn ein Objekt wird ja zurückgegeben, aber der Typ kann irgend etwas sein.
// Eclipse generiert als Resultattyp (falls der Typ der lokalen Variablen nicht angegeben wird) Animal, da dies der Bottom-Typ
// des Typparameters ist.
// Dieser Fall ist vergleichbar mit null als Input.
