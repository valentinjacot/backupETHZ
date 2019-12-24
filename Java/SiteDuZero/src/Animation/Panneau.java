package Animation;

import java.awt.Color;
import java.awt.Graphics;

import javax.swing.JPanel;

public class Panneau extends JPanel {
	 
    private int posX = -50;
    private int posY = -50;
 
    public void paintComponent(Graphics g) {
    	//comment the first 2 lines to have the trace following your annimation	
        // On décide d'une couleur de fond pour notre rectangle
        g.setColor(Color.white);
        // On dessine celui-ci afin qu'il prenne tout la surface
        g.fillRect(0, 0, this.getWidth(), this.getHeight());
        // On redéfinit une couleur pour notre rond
        g.setColor(Color.red);
        // On le dessine aux coordonnées souhaitées
        g.fillOval(posX, posY, 50, 50);
    }
 
    public int getPosX() {
        return posX;
    }
 
    public void setPosX(int posX) {
        this.posX = posX;
    }
 
    public int getPosY() {
        return posY;
    }
 
    public void setPosY(int posY) {
        this.posY = posY;
    }
 
}