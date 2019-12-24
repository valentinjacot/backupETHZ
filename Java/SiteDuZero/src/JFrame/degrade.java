package JFrame;

import java.awt.Color;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D;

import javax.swing.JPanel;

public class degrade extends JPanel {
	public void paintComponent(Graphics g) {
		Graphics2D g2d = (Graphics2D)g;
		System.out.println("je suis invoqué!");

		GradientPaint gp = new GradientPaint(0,0,Color.RED,30,30, Color.CYAN,true);
		g2d.setPaint(gp);
		g2d.fillRect(0, 0, this.getWidth(), this.getHeight());
		
	}
}

