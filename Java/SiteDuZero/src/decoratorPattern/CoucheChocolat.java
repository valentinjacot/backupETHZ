package decoratorPattern;

public class CoucheChocolat extends Couche{
	public CoucheChocolat(Patisserie p) {
		super(p);
		this.nom = "\t- une couche de chocolat \n";
	}
}
