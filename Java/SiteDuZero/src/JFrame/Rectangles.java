package JFrame;

import java.awt.Graphics;

import javax.swing.JPanel;


public class Rectangles extends JPanel{
	public void paintComponent(Graphics g) {
		g.drawRect(10, 10, 50, 60);
		g.fillRect(65, 65, 30, 40);
		g.drawRoundRect(100, 100, 50, 60, 10, 10);
	}
}
