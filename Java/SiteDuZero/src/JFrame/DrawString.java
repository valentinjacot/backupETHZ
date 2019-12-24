package JFrame;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;

import javax.swing.JPanel;

public class DrawString extends JPanel {
	public void paintComponent(Graphics g) {
		Font font = new Font("Courier", Font.BOLD, 20);
		g.setFont(font);
		g.setColor(Color.red);
		g.drawString("Hello World!!", 10, 20);
	}
}
