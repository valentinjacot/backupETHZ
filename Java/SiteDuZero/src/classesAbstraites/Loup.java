package classesAbstraites;

public class Loup extends Canin {
	  
    public Loup(){
             
    }
    public Loup(String couleur, int poids){
            this.couleur = couleur;
            this.poids = poids;
    }      

    void crier() {
            System.out.println("J'hurle à la lune en faisant ouhouh ! ! ");               
    }
}