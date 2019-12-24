package classesAbstraites;

public class Chien extends Canin implements Rintintin{
	  
    public Chien(){
             
    }
    public Chien(String couleur, int poids){
            this.couleur = couleur;
            this.poids = poids;
    }      

     
    void crier() {
            System.out.println("J'aboie sans raison ! ");
    }
	@Override
	public void faireCalin() {
        System.out.println("Je fais un calin");
	}
	@Override
	public void faireQqch() {
        System.out.println("Je fais qqch");
	}
	@Override
	public void faireLeBeau() {
        System.out.println("Je fais le beau");
	}

}