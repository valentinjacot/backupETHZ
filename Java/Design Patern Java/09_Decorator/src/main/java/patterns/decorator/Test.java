package patterns.decorator;

public class Test {
	
	public static void main(String[] args) {
		Phone phone = new Mobile();
		Phone redPhone = new RedDecorator(phone);
		
		redPhone.draw(null);
		redPhone.dial("+41561234567");
	}

}
