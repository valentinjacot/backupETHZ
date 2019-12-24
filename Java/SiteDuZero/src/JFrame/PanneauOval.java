package JFrame;

import java.awt.Graphics;

import javax.swing.JPanel;

public class PanneauOval extends JPanel{
	public void paintComponent(Graphics g) {
		System.out.println("je suis invoqué!");
		int x1 = this.getWidth()/4;
		int y1 = this.getHeight()/4;
		g.fillOval(x1, y1, this.getWidth()/2, this.getHeight()/2);
	}             
}