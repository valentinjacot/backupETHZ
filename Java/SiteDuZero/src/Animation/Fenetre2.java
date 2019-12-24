package Animation;

import java.awt.Dimension;
 
import javax.swing.JFrame;
 
public class Fenetre2 extends JFrame {
 
    private Panneau pan = new Panneau();
 
    public Fenetre2() {
 
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
 
        // Dans cet exemple, j'utilise une boucle while
        // Vous verrez qu'elle fonctionne tr�s bien
        while (true) {
            // Si la coordonn�e x est inf�rieure � 1, on avance
            if (x < 1)
                backX = false;
            // Si la coordonn�e x est sup�rieure � la taille du Panneau
            // moins la taille du rond, on recule
            if (x > pan.getWidth() - 50)
                backX = true;
            // Idem pour l'axe y
            if (y < 1)
                backY = false;
            if (y > pan.getHeight() - 50)
                backY = true;
 
            // Si on avance, on incr�mente la coordonn�e
            // backX est un bool�en, donc !backX revient � �crire
            // if (backX == false)
            if (!backX)
                pan.setPosX(++x);
            // Sinon, on d�cr�mente
            else
                pan.setPosX(--x);
            // Idem pour l'axe Y
            if (!backY)
                pan.setPosY(++y);
            else
                pan.setPosY(--y);
 
            // On redessine notre Panneau
            pan.repaint();
            // Comme on dit : la pause s'impose ! Ici, trois milli�mes de
            // seconde
            try {
                Thread.sleep(3);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}