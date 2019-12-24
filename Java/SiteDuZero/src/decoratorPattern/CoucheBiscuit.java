package decoratorPattern;

public class CoucheBiscuit extends Couche{
	public CoucheBiscuit (Patisserie p) {
		super(p);
		this.nom = "\t- une couche de biscuit \n";
	}
}
