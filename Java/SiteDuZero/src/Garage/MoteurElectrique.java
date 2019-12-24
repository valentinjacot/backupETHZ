package Garage;

public class MoteurElectrique extends Moteur {
	public MoteurElectrique(String cv, double prix_) {
		super(cv,prix_);
		this.type = TypeMoteur.ELECTIRIQUE;
		//this.prix = prix_;
	}

}
