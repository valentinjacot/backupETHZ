package Decorators;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Pizza Margarita= new Mozzarella(new TomatoSauce(new PlainPizza()));
		System.out.println(Margarita);
		System.out.println("And costs: " + Margarita.getCost());
		
		Pizza HamPizza= new Ham(new Mozzarella(new TomatoSauce(new PlainPizza())));
		System.out.println(HamPizza);
		System.out.println("And costs: " + HamPizza.getCost());
		
		Pizza Nature = new TomatoSauce(new PlainPanini());
		System.out.println(Nature);
		System.out.println("And costs: " + Nature.getCost());
		
		Pizza MozPan = new Mozzarella(new TomatoSauce(new PlainPanini()));
		System.out.println(MozPan);
		System.out.println("And costs: " + MozPan.getCost());
	}

}
