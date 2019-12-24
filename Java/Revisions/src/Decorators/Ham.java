package Decorators;

public class Ham  extends AbstractDecorator{

	public Ham(Pizza p) {
		super(p);
	}

	@Override
	public String toString() {
		return super.tempPizza.toString() + ", Ham";
	}

	@Override
	public double getCost() {
		return  super.tempPizza.getCost() + 2.5;
	}

}
