package JFrame;

import java.awt.Color;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Graphics2D; 
import javax.imageio.ImageIO;
import javax.swing.JPanel;
 
public class arcEnCiel extends JPanel { 
  public void paintComponent(Graphics g){
    Graphics2D g2d = (Graphics2D)g;
    GradientPaint gp, gp2, gp3, gp4, gp5, gp6; 
    gp = new GradientPaint(0, 0, Color.RED, 20, 0, Color.magenta, true);
    gp2 = new GradientPaint(20, 0, Color.magenta, 40, 0, Color.blue, true);
    gp3 = new GradientPaint(40, 0, Color.blue, 60, 0, Color.green, true);
    gp4 = new GradientPaint(60, 0, Color.green, 80, 0, Color.yellow, true);
    gp5 = new GradientPaint(80, 0, Color.yellow, 100, 0, Color.orange, true);
    gp6 = new GradientPaint(100, 0, Color.orange, 120, 0, Color.red, true);

    g2d.setPaint(gp);
    g2d.fillRect(0, 0, 20, this.getHeight());               
    g2d.setPaint(gp2);
    g2d.fillRect(20, 0, 20, this.getHeight());
    g2d.setPaint(gp3);
    g2d.fillRect(40, 0, 20, this.getHeight());
    g2d.setPaint(gp4);
    g2d.fillRect(60, 0, 20, this.getHeight());
    g2d.setPaint(gp5);
    g2d.fillRect(80, 0, 20, this.getHeight());
    g2d.setPaint(gp6);
    g2d.fillRect(100, 0, 40, this.getHeight());
  }               
}