package Garage;

public class MoteurHybride extends Moteur{
	public MoteurHybride(String cv, double prix_) {
		super(cv,prix_);
		this.type = TypeMoteur.HYBRID;
		//this.prix = prix_;
	}

}
