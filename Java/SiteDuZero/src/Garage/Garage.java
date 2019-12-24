package Garage;

import java.util.ArrayList;

public class Garage {
	private ArrayList<Vehicule> voitures = new ArrayList<Vehicule>();
	public Garage() {
	}
	public Garage(ArrayList<Vehicule> voit) {
		this.voitures = voit;
	}
	public String toString() {
		String str ="*************************\n";
		str += "* Garage OpenClassrooms *\n";
		str += "*************************\n";
		if (voitures.isEmpty()) 
			str += "Le garage est vide";
		else
		{
			for (Vehicule voit:voitures) {
				str += voit.toString()+ "\n";
			}
		}
		return str;
	}

	public void addVoiture(Vehicule voit) {
		voitures.add(voit);
	}

}
