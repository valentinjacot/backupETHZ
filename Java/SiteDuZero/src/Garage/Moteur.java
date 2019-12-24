package Garage;

public class Moteur {
	protected TypeMoteur type;
	protected double prix;
	protected String cylindre;
	public Moteur(String cyl_, double prix_) {
		this.cylindre = cyl_;
		this.prix = prix_;
	}
	public String toString() {
		return " Moteur " + type + " " + cylindre ; //+ "(" + prix + "$)";
	}
	
	public double getPrix(){
		return prix;
	}

}
