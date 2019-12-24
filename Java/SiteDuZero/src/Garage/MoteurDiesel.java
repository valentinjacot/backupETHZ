package Garage;

public class MoteurDiesel extends Moteur{
	public MoteurDiesel(String cv, double prix_) {
		super(cv,prix_);
		this.type = TypeMoteur.DIESEL;
		//this.prix = prix_;
	}

}
