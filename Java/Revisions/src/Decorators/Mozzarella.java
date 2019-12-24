package Decorators;

public class Mozzarella extends AbstractDecorator{

	public Mozzarella(Pizza p) {
		super(p);
	}

	@Override
	public String toString() {
		return tempPizza.toString() + ", Morarella";
	}

	@Override
	public double getCost() {
		return  tempPizza.getCost() + 1.5;
	}

}
