package classesAbstraites;

public class Test {
	public static void main(String[] args) {
		Chien l = new Chien("Gris bleu", 20 );
		l.boire();
		l.manger();
		l.deplacement();
		l.crier();
		System.out.println(l.toString());
		l.faireLeBeau();
		l.faireCalin();
		l.faireQqch();
		
		Rintintin r = new Chien();
		r.faireLeBeau();
		r.faireCalin();
	}

}
