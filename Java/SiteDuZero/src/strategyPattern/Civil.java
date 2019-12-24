package strategyPattern;

import comportement.Deplacement;
import comportement.EspritCombatif;
import comportement.Soin;

public class Civil extends Personnage{
	public Civil() {};
	public Civil(EspritCombatif espritCombatif, Soin soin,
            Deplacement deplacement) {
		super(espritCombatif, soin,deplacement);
	}
}