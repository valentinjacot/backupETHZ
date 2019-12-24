package Builder;

public class Pizza {
	private double cost;
	private Boolean Mozzarela;
	private Boolean Peperoni;
	private Boolean Bacon;

	Pizza(Builder p) {
		this.cost = p.cost;
		this.Mozzarela = p.Mozzarela;
		this.Bacon= p.Bacon;
		this.Peperoni=p.Peperoni;
	}
	
	public double getCost() {
		return cost;
	}
	public void setCost(double cost) {
		this.cost = cost;
	}
	
	@Override
	public String toString() {
		return "Pizza [cost=" + cost + ", Mozzarela=" + Mozzarela + ", Peperoni=" + Peperoni + ", Bacon=" + Bacon + "]";
	}
	
}
