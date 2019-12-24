package Decorators;

public class PlainPanini implements Pizza {
	private String description;
	private double cost;
	public PlainPanini() {
		this.description="";
		this.cost = 3.0;
	}
	@Override
	public double getCost() {
		// TODO Auto-generated method stub
		return cost;
	}
	@Override
	public String toString() {
		return "This Panino is made of Bread" + description;
	}
	

}
