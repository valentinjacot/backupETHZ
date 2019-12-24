package decoratorPattern;

public class CoucheCaramel extends Couche{
	public CoucheCaramel(Patisserie p) {
		super(p);
		this.nom = "\t- une couche de caramel \n";
	}
}