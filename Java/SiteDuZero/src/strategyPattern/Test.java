package strategyPattern;

import comportement.*;

public class Test {
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Personnage[] tPers= {new Guerrier(), new Chirurgien(), new Civil(), new Sniper(), new Medecin()};
		for (Personnage p : tPers) {
			System.out.println("\n instance de " + p.getClass().getName());
			System.out.println("**************");
			p.combattre();
			p.seDeplacer();
			p.soigner();
		}
		System.out.println("*****************************************");
		
		Personnage pers= new Guerrier(new CombatPistolet(), new AucunSoin(), new Courir());
		pers.soigner();
		pers.setSoin(new Operation());
		pers.soigner();
		pers.seDeplacer();
		pers.combattre();
//		 //Personnage[] tPers = {new Guerrier(), new Chirurgien(), new Civil(), new Sniper(), new Medecin()};
//	        String[] tArmes = {"pistolet", "pistolet", "couteau", "fusil à pompe", "couteau"};
//	        for(int i = 0; i < tPers.length; i++){
//	            System.out.println("\nInstance de " + tPers[i].getClass().getName());
//	            System.out.println("*****************************************");
//	            tPers[i].combattre();
//	            tPers[i].seDeplacer();
//	            tPers[i].soigner();
//	        }      
		}

}
