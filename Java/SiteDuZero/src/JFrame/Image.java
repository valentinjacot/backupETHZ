package JFrame;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JPanel;

public class Image extends JPanel {
	public void paintComponent(Graphics g) {
		try {
			BufferedImage img = ImageIO.read(new File("trump_gay.jpg"));
			g.drawImage(img, 0, 0, this);
		}catch(IOException e) {
			e.printStackTrace();
		}
	}
}