package Builder;


public class Builder{
	double cost;
	Boolean Mozzarela;
	Boolean Peperoni;
	Boolean Bacon;
	public Builder() {
		this.cost = 4;//base cost
		this.Mozzarela=false;
		this.Bacon=false;
		this.Peperoni=false;
	}
	public Builder setMozz() {
		this.Mozzarela =true;
		this.cost += 1;
		return this;
	}
	public Builder setPep() {
		this.Peperoni =true;
		this.cost += 0.5;
		return this;
	}
	public Builder setBac() {
		this.Bacon =true;
		this.cost += 2;
		return this;
	}
	public Pizza build() {
		return new Pizza(this);
	}
}
