package strategyPattern;
import comportement.*;
public abstract class Personnage {
	 
    //Nos instances de comportements
    protected EspritCombatif espritCombatif = new Pacifiste();
    protected Soin soin = new AucunSoin();
    protected Deplacement deplacement = new Marcher();
     
    /**
     * Constructeur par d�faut
     */
    public Personnage(){}
     
    /**
     * Constructeur avec param�tres
     * @param espritCombatif
     * @param soin
     * @param deplacement
     */
    public Personnage(EspritCombatif espritCombatif, Soin soin,
            Deplacement deplacement) {
        this.espritCombatif = espritCombatif;
        this.soin = soin;
        this.deplacement = deplacement;
    }
    /**
     * M�thode de d�placement de personnage
     */
    public void seDeplacer(){
        //On utilise les objets de d�placement de fa�on polymorphe
        deplacement.deplacer();
    }
    /**
     * M�thode que les combattants utilisent
     */
    public void combattre(){
        //On utilise les objets de d�placement de fa�on polymorphe
        espritCombatif.combat();
    }
    /**
     * M�thode de soin
     */
    public void soigner(){
        //On utilise les objets de d�placement de fa�on polymorphe
        soin.soigne();
    }
     
    //************************************************************
    //                      ACCESSEURS
    //************************************************************
     
    /**
     * Red�finit le comportement au combat
     * @param espritCombatif
     */
    protected void setEspritCombatif(EspritCombatif espritCombatif) {
        this.espritCombatif = espritCombatif;
    }
    /**
     * Red�finit le comportement de Soin
     * @param soin
     */
    protected void setSoin(Soin soin) {
        this.soin = soin;
    }
    /**
     * Red�finit le comportement de d�placement
     * @param deplacement
     */
    protected void setDeplacement(Deplacement deplacement) {
        this.deplacement = deplacement;
    }  
}