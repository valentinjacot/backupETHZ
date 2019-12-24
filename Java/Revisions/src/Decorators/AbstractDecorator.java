package Decorators;

public abstract class AbstractDecorator implements Pizza{
	
	protected Pizza tempPizza;
	protected AbstractDecorator(Pizza p) {
		tempPizza=p;
	}
	@Override
	public abstract String toString();
	public abstract double getCost();
	
	
}
