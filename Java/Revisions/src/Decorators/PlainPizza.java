package Decorators;

public class PlainPizza implements Pizza{
	private double cost;
	private String description;
	
	public PlainPizza() {
		cost=4.0;
		description = "";
	}
	@Override
	public double getCost() {
		return cost;
	}
	
	@Override
	public String toString() {
		return "This Pizza is made of dough" + description;
	}
	
}
