package Decorators;

public class TomatoSauce extends AbstractDecorator{

	public TomatoSauce(Pizza p) {
		super(p);
	}

	@Override
	public String toString() {
		return super.tempPizza.toString() + ", Tomato Sauce";
	}

	@Override
	public double getCost() {
		return  super.tempPizza.getCost() + 1.0;
	}

}
