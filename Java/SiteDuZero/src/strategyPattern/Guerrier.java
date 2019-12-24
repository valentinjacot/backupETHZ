package strategyPattern;

import comportement.*;
public class Guerrier extends Personnage {
	public Guerrier() {
		this.espritCombatif = new CombatPistolet();
	};
	public Guerrier(EspritCombatif espritCombatif, Soin soin,
            Deplacement deplacement) {
		super(espritCombatif, soin,deplacement);
	}

}