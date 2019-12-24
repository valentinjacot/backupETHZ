package Builder;

public class Main {

	public static void main(String[] args) {
		Pizza pizza= new  Builder().setPep().setBac().setMozz().build();
		System.out.println(pizza);
	}

}
