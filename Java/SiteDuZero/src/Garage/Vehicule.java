package Garage;

import java.util.ArrayList;

public class Vehicule {
	protected double prixCatalogue;
	private double prix_tot;
	protected String nom;
	protected ArrayList<Option> options = new ArrayList<Option>();
	protected Marque nomMarque;
	protected Moteur moteur;
	
	
	public String toString() {
		String optName= " [";
		for(Option opt:options) {
			optName+= opt.toString() + " (" + opt.getPrix() + "), ";
		}
		optName += "]";
		return "+ Voiture " + nomMarque + " : " + nom + moteur.toString() + "(" + prixCatalogue+ "$)" + optName + " d'une valeur totale de " + (prix_tot + prixCatalogue) + "$";
	}
	public void addOption(Option opt) {
		options.add(opt);
		this.prix_tot += opt.getPrix();
	}
	
	public Marque getMarque() {
		return nomMarque;
	}
	public void setMoteur(Moteur mot_) {
		// TODO Auto-generated method stub
		this.moteur= mot_;
		//this.prix_tot += mot_.getPrix();
	}
	
	
}
