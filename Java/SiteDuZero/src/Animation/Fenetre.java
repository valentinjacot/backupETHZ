package Animation;

import javax.swing.JFrame;

public class Fenetre extends JFrame{
	private Panneau pan = new Panneau();

	public Fenetre() {
		this.setTitle("Animation");
		this.setSize(300, 300);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setLocationRelativeTo(null);
		this.setContentPane(pan);
		this.setVisible(true);
		
		go();
	}

	private void go() {
		// Les coordonn�es de d�part de notre rond
		int x = pan.getPosX(), y = pan.getPosY();
		// Le bool�en pour savoir si l'on recule ou non sur l'axe x
		boolean backX = false;
		// Le bool�en pour savoir si l'on recule ou non sur l'axe y
		boolean backY = false;
		
		
		while(true) {
			// Si la coordonn�e x est inf�rieure � 1, on avance
			if (x < 1) 
				backX=false;
			// Si la coordonn�e x est sup�rieure � la taille du Panneau
			// moins la taille du rond, on recule
			if(x > pan.getWidth() - 50) 
				backX = true;
			// Idem pour l'axe y
			if (y < 1) 
				backY=false;
			if (y > pan.getHeight() - 50) 
				backY=true;
			// Si on avance, on incr�mente la coordonn�e
			// backX est un bool�en, donc !backX revient � �crire
			// if (backX == false)
			if(!backX) 
				pan.setPosX(++x);
			else 
				pan.setPosX(--x);
			if(!backY) 
				pan.setPosY(++y);
			else 
				pan.setPosY(--y);
			//			x++;
			//			y++;
			//			pan.setPosX(x);
			//			pan.setPosY(y);
			pan.repaint();
			
			try {
				Thread.sleep(3);
			}catch(InterruptedException e) {
				e.printStackTrace();
			}
			//			if(x==pan.getWidth() || y==pan.getHeight()) {
			//				pan.setPosX(-50);
			//				pan.setPosY(-50);
			//			}
		}
	}
}
