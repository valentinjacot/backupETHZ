package Garage;

public class MoteurEssence extends Moteur{
	public MoteurEssence(String cv, double prix_) {
		super(cv,prix_);
		this.type = TypeMoteur.ESSENCE;
		//this.prix = prix_;
	}
}
