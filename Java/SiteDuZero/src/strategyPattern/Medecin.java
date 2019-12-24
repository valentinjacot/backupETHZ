package strategyPattern;

import comportement.*;

public class Medecin extends Personnage{

	public Medecin() {
		this.soin= new PremierSoin();
	};
	public Medecin(EspritCombatif espritCombatif, Soin soin,
            Deplacement deplacement) {
		super(espritCombatif, soin,deplacement);
	}
}