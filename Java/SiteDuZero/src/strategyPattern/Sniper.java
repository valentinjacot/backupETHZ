package strategyPattern;

import comportement.*;

public class Sniper extends Personnage{
	public Sniper() {
		this.espritCombatif = new CombatSniper();
	};
	public Sniper(EspritCombatif espritCombatif, Soin soin,
            Deplacement deplacement) {
		super(espritCombatif, soin,deplacement);
	}
}